import time
import random
import sys

N = 128
DEADLINE = 85.0  # safety margin before 90s

def construct():
    start = time.time()
    
    # Precompute all nonconstant 4-term progressions
    # A 4-term progression is (a, a+d, a+2d, a+3d) with d > 0
    # Positions are 0-indexed: 0 to 127
    progressions = []
    for a in range(N):
        for d in range(1, (N - 1 - a) // 3 + 1):
            if a + 3 * d < N:
                progressions.append((a, a + d, a + 2 * d, a + 3 * d))
    
    num_progs = len(progressions)
    
    # We want to minimize the number of distinct colors used.
    # The seed uses a greedy approach that tends to use many colors.
    # Key insight: we need at least 3 colors in every 4-term AP.
    # A 2-coloring is impossible for this constraint on [1,128] for large enough N.
    # Let's try to find a coloring using as few colors as possible.
    # Minimum possible is 3 (since we need at least 3 distinct colors in each AP).
    
    # Try 3 colors first: 0, 1, 2
    # Use backtracking with constraint propagation
    
    def try_k_colors(k, seed=0):
        """Try to find a valid coloring using exactly k colors (labels 0..k-1)."""
        rng = random.Random(seed)
        colors = [-1] * N
        colors[0] = 0
        
        # For each position, track which progressions it's part of
        pos_to_progs = [[] for _ in range(N)]
        for idx, prog in enumerate(progressions):
            for p in prog:
                pos_to_progs[p].append(idx)
        
        # Track progressions that are already satisfied (have >= 3 colors)
        prog_satisfied = [False] * num_progs
        # Track for each progression, the set of colors used
        prog_colors = [set() for _ in range(num_progs)]
        
        # Order positions for backtracking
        order = list(range(N))
        
        def check_progs(pos, color):
            """Check if assigning color to pos is consistent with all progressions through pos."""
            for idx in pos_to_progs[pos]:
                prog = progressions[idx]
                if prog_colors[idx]:
                    # If this progression already has the color, that's fine
                    # We just need at least 3 distinct colors total
                    pass
            
            # Check: for any progression through pos where all other 3 positions
            # are already assigned and use only 2 colors, and color is one of those 2,
            # it's still OK (we need at least 3). But if all 4 would use only 2 colors, bad.
            for idx in pos_to_progs[pos]:
                prog = progressions[idx]
                others = [p for p in prog if p != pos]
                other_colors = set()
                all_assigned = True
                for o in others:
                    if colors[o] == -1:
                        all_assigned = False
                        break
                    other_colors.add(colors[o])
                if all_assigned:
                    new_set = other_colors | {color}
                    if len(new_set) < 3:
                        return False
            return True
        
        def backtrack(pos_idx):
            """Backtracking search. pos_idx is the index in the order list."""
            if time.time() - start > DEADLINE:
                return False
            if pos_idx == N:
                return True
            
            pos = order[pos_idx]
            
            # Try colors in a smart order
            candidate_colors = list(range(k))
            rng.shuffle(candidate_colors)
            
            for c in candidate_colors:
                colors[pos] = c
                
                # Check consistency
                valid = True
                for idx in pos_to_progs[pos]:
                    prog = progressions[idx]
                    prog_set = set()
                    for p in prog:
                        if colors[p] == -1:
                            continue
                        prog_set.add(colors[p])
                    if len(prog_set) < 3:
                        # Check if it's possible to still reach 3 colors
                        unassigned_in_prog = [p for p in prog if colors[p] == -1]
                        if not unassigned_in_prog and len(prog_set) < 3:
                            valid = False
                            break
                        # Even with unassigned, if current colors < 3 and unassigned can't add enough
                        # This is complex; simple check: if all assigned and < 3, fail
                        if not unassigned_in_prog:
                            valid = False
                            break
                        # If we have 2 colors assigned and 1 unassigned, the unassigned must be a 3rd color
                        # If we have 1 color assigned and 2 unassigned, at least one must differ
                        # Simple: just check if currently assigned < 3 and no unassigned can fix it
                        if len(prog_set) < 3:
                            # Count distinct colors among assigned
                            # If 0 or 1 unassigned positions remain, and we can't add enough new colors
                            remaining_unassigned = len(unassigned_in_prog)
                            current_distinct = len(prog_set)
                            # We need current_distinct + (max possible new colors from unassigned) >= 3
                            # Max new colors = min(remaining_unassigned, k - current_distinct)
                            if current_distinct + min(remaining_unassigned, k - current_distinct) < 3:
                                valid = False
                                break
                
                if not valid:
                    colors[pos] = -1
                    continue
                
                # Check: for any progression through pos, if all 4 positions are now assigned
                # and they use < 3 colors, invalid
                for idx in pos_to_progs[pos]:
                    prog = progressions[idx]
                    prog_set = set()
                    all_ass = True
                    for p in prog:
                        if colors[p] == -1:
                            all_ass = False
                            break
                        prog_set.add(colors[p])
                    if all_ass and len(prog_set) < 3:
                        valid = False
                        break
                
                if not valid:
                    colors[pos] = -1
                    continue
                
                if backtrack(pos_idx + 1):
                    return True
                
                colors[pos] = -1
            
            return False
        
        return backtrack(1)  # Position 0 already assigned
    
    # Try 3 colors with several random seeds
    for seed in range(50):
        if time.time() - start > DEADLINE - 5:
            break
        colors = [-1] * N
        colors[0] = 0
        rng = random.Random(seed)
        
        # Use a more efficient approach: simulate and check
        # Actually, let's use a different strategy: 
        # Start with a base pattern and repair violations
        
        # Pattern: use 3 colors in a structured way
        # color[i] = i % 3? Let's check if this works.
        # AP: a, a+d, a+2d, a+3d. Colors: a%3, (a+d)%3, (a+2d)%3, (a+3d)%3
        # If d % 3 == 0: all same color -> BAD
        # So i%3 doesn't work.
        
        # Try: color based on 2-bit representation or similar
        # Let's try a different approach: use the fact that we need 3 colors per AP.
        # A known result: 3 colors suffice for 3-term APs. For 4-term APs, 
        # we need to check if 3 colors suffice.
        
        # Let me try: color[i] = (i // 2) % 3 for even, and (i // 2) % 3 + something for odd?
        # Actually, let's just do a proper search.
        
        # Use greedy with lookahead: assign colors to positions in order,
        # choosing the color that minimizes future conflicts.
        
        def greedy_with_k(k, seed_val):
            rng = random.Random(seed_val)
            colors = [-1] * N
            colors[0] = 0
            
            # For each position, we'll choose a color
            for pos in range(1, N):
                if time.time() - start > DEADLINE - 5:
                    return None
                
                # Find all progressions that include this position
                # and have some positions already assigned
                best_color = -1
                best_score = -1
                
                for c in range(k):
                    colors[pos] = c
                    
                    # Check all progressions through pos
                    valid = True
                    conflict_count = 0
                    for idx in pos_to_progs[pos]:
                        prog = progressions[idx]
                        prog_set = set()
                        unassigned = 0
                        for p in prog:
                            if colors[p] == -1:
                                unassigned += 1
                            else:
                                prog_set.add(colors[p])
                        
                        if unassigned == 0:
                            # All assigned: need >= 3 colors
                            if len(prog_set) < 3:
                                valid = False
                                break
                        else:
                            # Check if it's still possible
                            current_distinct = len(prog_set)
                            # Max additional distinct colors from unassigned
                            max_add = min(unassigned, k - current_distinct)
                            if current_distinct + max_add < 3:
                                valid = False
                                break
                            # Count how "tight" this is
                            if current_distinct + max_add == 3:
                                conflict_count += 1
                    
                    colors[pos] = -1
                    if not valid:
                        continue
                    
                    # Score: prefer colors that leave more flexibility
                    # Lower conflict_count is better
                    score = -conflict_count
                    if score > best_score:
                        best_score = score
                        best_color = c
                
                if best_color == -1:
                    # No valid color found, try with random perturbation
                    # or this k doesn't work
                    return None
                
                colors[pos] = best_color
            
            # Verify the complete coloring
            for prog in progressions:
                prog_set = set(colors[p] for p in prog)
                if len(prog_set) < 3:
                    return None
            
            return colors
        
        # Try 3 colors
        for seed_val in range(100):
            if time.time() - start > DEADLINE - 5:
                break
            result = greedy_with_k(3, seed_val)
            if result is not None:
                return result
        
        # Try 4 colors if 3 doesn't work quickly
        for seed_val in range(50):
            if time.time() - start > DEADLINE - 5:
                break
            result = greedy_with_k(4, seed_val)
            if result is not None:
                return result
        
        # Try 5 colors
        for seed_val in range(30):
            if time.time() - start > DEADLINE - 5:
                break
            result = greedy_with_k(5, seed_val)
            if result is not None:
                return result
    
    # Fallback: use the original greedy approach
    colors = []
    for x in range(1, 129):
        if time.time() - start > DEADLINE - 1:
            # Fill rest with a safe pattern
            for _ in range(128 - len(colors)):
                colors.append(0)
            break
        found = False
        for color in range(128):
            valid = True
            # Check all 4-term APs ending at x-1
            for d in range(1, (x - 1) // 3 + 1):
                if x - 1 - 3 * d < 0:
                    break
                ap_colors = {color}
                for j in range(1, 4):
                    ap_colors.add(colors[x - 1 - j * d])
                if len(ap_colors) < 3:
                    valid = False
                    break
            # Check all 4-term APs where x-1 is at position 1 (a+d)
            if valid:
                for d in range(1, (x - 1) // 2 + 1):
                    if x - 1 - d < 0 or x - 1 + d >= 128:
                        break
                    # AP: x-1-d, x-1, x-1+d, x-1+2d
                    if x - 1 + 2 * d >= 128:
                        break
                    # We can only check if all are assigned
                    if x - 1 + 2 * d <= x - 1:
                        continue
                    # Future positions not assigned yet, skip for now
                    pass
            if valid:
                colors.append(color)
                found = True
                break
        if not found:
            colors.append(0)
    
    # Verify and fallback if invalid
    for prog in progressions:
        prog_set = set(colors[p] for p in prog)
        if len(prog_set) < 3:
            # This is bad; force a valid coloring with many colors
            # Use position-based coloring: color[i] = i (all distinct)
            # That uses 128 colors but is valid
            colors = list(range(128))
            break
    
    return colors
