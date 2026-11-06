"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product

def _precompute_aps():
    """Precompute all nonconstant 4-term APs in [1,128]."""
    n = 128
    aps = []
    for a in range(1, n + 1):
        for d in range(1, n):
            b = a + d
            c = a + 2*d
            e = a + 3*d
            if e <= n:
                # Store 0-indexed
                aps.append((a-1, b-1, c-1, e-1))
    return aps

def _build_degree_order(aps):
    """Order positions by degree (number of 4-APs they appear in)."""
    n = 128
    degree = [0] * n
    for ap in aps:
        for pos in ap:
            degree[pos] += 1
    # Sort by descending degree
    order = sorted(range(n), key=lambda x: -degree[x])
    return order, degree

def _build_ap_lookup(aps):
    """For each position, list of (ap_index, other_positions) for quick checking."""
    n = 128
    pos_aps = [[] for _ in range(n)]
    for i, ap in enumerate(aps):
        for pos in ap:
            others = [p for p in ap if p != pos]
            pos_aps[pos].append((i, others))
    return pos_aps

def _count_distinct_colors(assignment, positions):
    """Count distinct colors among assigned positions in the list."""
    colors = set()
    for p in positions:
        if assignment[p] is not None:
            colors.add(assignment[p])
    return len(colors)

def _can_assign_color(assignment, pos, color, pos_aps, aps):
    """Check if assigning color to pos doesn't immediately violate any 4-AP."""
    for ap_idx, others in pos_aps[pos]:
        ap = aps[ap_idx]
        # Check if this AP now has all 4 positions assigned
        all_assigned = all(assignment[p] is not None for p in ap)
        if all_assigned:
            distinct = len({assignment[p] for p in ap})
            if distinct < 3:
                return False
        # Also check: if 3 positions are assigned and use only 1 or 2 colors,
        # and the 4th is unassigned, we need to ensure it CAN be different
        # For forward checking: if 3 assigned positions use only 1 color,
        # the 4th must be different (which is always possible with 2+ colors)
        # If 3 assigned use 2 colors, the 4th can be either - no constraint violated yet
        # The real constraint is: we need AT LEAST 3 distinct colors total
        # So if 3 are assigned and use only 1 distinct color, the 4th MUST be a new color
        # If 3 are assigned and use 2 distinct colors, the 4th can be any of the 2 or a new one
        # Violation only when all 4 assigned and < 3 distinct
    return True

def _forward_check_violation(assignment, pos, color, pos_aps, aps):
    """After assigning pos=color, check if any unassigned position is forced to violate."""
    for ap_idx, others in pos_aps[pos]:
        ap = aps[ap_idx]
        assigned_colors = set()
        unassigned = []
        for p in ap:
            if assignment[p] is not None:
                assigned_colors.add(assignment[p])
            else:
                unassigned.append(p)
        
        if len(unassigned) == 0:
            # All assigned - check validity
            if len(assigned_colors) < 3:
                return True
        elif len(unassigned) == 1:
            # One position unassigned - it must make the total >= 3 distinct
            # Current distinct colors from assigned
            if len(assigned_colors) >= 3:
                pass  # Already satisfied
            elif len(assigned_colors) == 2:
                # The unassigned one can be either of the 2 colors or a new one
                # No violation possible since it can pick a 3rd color
                pass
            elif len(assigned_colors) == 1:
                # The unassigned one MUST be a different color
                # This is a constraint but not a violation yet
                pass
            else:
                # No assigned colors in this AP besides pos? Can't happen since pos is assigned
                pass
        else:
            # Multiple unassigned - no immediate violation
            pass
    return False

def _backtrack_search(aps, pos_aps, order, num_colors, deadline):
    """Backtracking search to find a valid coloring with num_colors colors."""
    n = 128
    assignment = [None] * n
    
    # Try colors in a fixed order for determinism
    color_list = list(range(num_colors))
    
    def backtrack(idx):
        if time.time() > deadline:
            return None
        
        if idx == n:
            return assignment[:]
        
        pos = order[idx]
        
        # Try each color
        for color in color_list:
            # Check if this assignment is locally valid
            if not _can_assign_color(assignment, pos, color, pos_aps, aps):
                continue
            
            # Assign
            assignment[pos] = color
            
            # Forward check
            if _forward_check_violation(assignment, pos, color, pos_aps, aps):
                assignment[pos] = None
                continue
            
            # Recurse
            result = backtrack(idx + 1)
            if result is not None:
                return result
            
            assignment[pos] = None
        
        return None
    
    return backtrack(0)

def _validate_coloring(colors, aps):
    """Check if the coloring is valid."""
    for ap in aps:
        distinct = len({colors[p] for p in ap})
        if distinct < 3:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit
    
    aps = _precompute_aps()
    order, degree = _build_degree_order(aps)
    pos_aps = _build_ap_lookup(aps)
    
    # Try with increasing number of colors
    for num_colors in [3, 4, 5, 6, 7, 8]:
        if time.time() > deadline - 5.0:
            break
        
        result = _backtrack_search(aps, pos_aps, order, num_colors, deadline)
        if result is not None:
            # Validate
            if _validate_coloring(result, aps):
                return result
    
    # Fallback: use the greedy approach from the original
    colors = []
    for x in range(1, 129):
        for color in range(128):
            valid = True
            # Check 4-APs ending at x
            for d in range(1, (x-1)//3 + 1):
                a = x - 1 - 3*d
                b = x - 1 - 2*d
                c = x - 1 - d
                if a < 0 or b < 0 or c < 0:
                    continue
                if len({color} | {colors[a], colors[b], colors[c]}) < 3:
                    valid = False
                    break
            if not valid:
                continue
            # Check 4-APs where x is not the last term
            for d in range(1, (x-1)//2 + 1):
                # x could be at position 1, 2, or 3 in the AP
                # Position 1: x, x+d, x+2d, x+3d - x+3d must be <= 128
                # Position 2: x-d, x, x+d, x+2d
                # Position 3: x-2d, x-d, x, x+d
                # We need to check that adding color doesn't cause violation
                # For simplicity, just check if it creates a 2-color AP
                pass  # This greedy is approximate; the backtracking should handle it
            if valid:
                colors.append(color)
                break
        else:
            # Should not happen with 128 colors
            colors.append(0)
    
    return colors

# EVOLVE-BLOCK-END
