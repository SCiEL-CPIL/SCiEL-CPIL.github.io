"""
Edge-colourings of K16 in which every K4 sees five colours.

Strategy:
1. Start from a structured or random colouring.
2. Use local search with O(1) incremental violation updates to fix violations.
3. Use local search to minimize the number of distinct colours used.
4. Return the best valid colouring found before the deadline.

Key optimization: Instead of re-evaluating all 1820 K4s after each move,
we precompute which K4s each edge belongs to and maintain per-K4 colour
count arrays, updating only the affected K4s in O(1) amortized time.
"""

import time
import random
import sys
import numpy as np

DEADLINE = 115.0  # safety margin before 120s kill

def _edge_index(i, j):
    """Map pair (i,j), 0<=i<j<16, to index in the 120-element list."""
    return i * 15 - i * (i - 1) // 2 + j - i - 1

def _precompute_k4s():
    """Return list of 6-tuples of edge indices for each K4."""
    k4s = []
    for a in range(16):
        for b in range(a + 1, 16):
            for c in range(b + 1, 16):
                for d in range(c + 1, 16):
                    edges = [
                        _edge_index(a, b),
                        _edge_index(a, c),
                        _edge_index(a, d),
                        _edge_index(b, c),
                        _edge_index(b, d),
                        _edge_index(c, d),
                    ]
                    k4s.append(tuple(edges))
    return k4s

_K4S = _precompute_k4s()
_K4S_ARRAY = np.array(_K4S, dtype=np.int32)  # shape (1820, 6)
_NUM_K4S = len(_K4S)

# Precompute for each edge, which K4s it belongs to
# edge_to_k4s[edge_idx] = list of K4 indices containing that edge
_EDGE_TO_K4S = [[] for _ in range(120)]
for k4_idx, k4_edges in enumerate(_K4S):
    for e in k4_edges:
        _EDGE_TO_K4S[e].append(k4_idx)

def _count_violations_fast(colours_arr):
    """Vectorized violation count using numpy. colours_arr is a numpy array of shape (120,)."""
    k4_colours = colours_arr[_K4S_ARRAY]  # (1820, 6)
    sorted_cols = np.sort(k4_colours, axis=1)  # (1820, 6)
    changes = (sorted_cols[:, 1:] != sorted_cols[:, :-1]).astype(np.int32)
    distinct_counts = 1 + changes.sum(axis=1)
    violations = np.sum(distinct_counts < 5)
    return int(violations)

def _num_colours_used(colours):
    return len(set(colours))

class _IncrementalTracker:
    """Maintains per-K4 colour counts for O(1) incremental updates."""
    
    def __init__(self, colours):
        self.colours = list(colours)
        # For each K4, track colour -> count
        self.k4_colour_counts = []
        self.violations = 0
        
        for k4_edges in _K4S:
            counts = {}
            for e in k4_edges:
                c = self.colours[e]
                counts[c] = counts.get(c, 0) + 1
            self.k4_colour_counts.append(counts)
            if len(counts) < 5:
                self.violations += 1
    
    def _update_k4(self, k4_idx, old_colour, new_colour):
        """Update a single K4 when an edge colour changes from old to new."""
        counts = self.k4_colour_counts[k4_idx]
        
        # Remove old colour
        old_count = counts.get(old_colour, 0)
        if old_count <= 1:
            del counts[old_colour]
            if len(counts) < 5:
                self.violations += 1
        else:
            counts[old_colour] = old_count - 1
        
        # Add new colour
        new_count = counts.get(new_colour, 0)
        if new_count == 0:
            if len(counts) < 5:
                self.violations -= 1
            counts[new_colour] = 1
        else:
            counts[new_colour] = new_count + 1
    
    def change_edge(self, edge_idx, new_colour):
        """Change an edge's colour and return the change in violations."""
        old_colour = self.colours[edge_idx]
        if old_colour == new_colour:
            return 0
        
        # Update all K4s containing this edge
        for k4_idx in _EDGE_TO_K4S[edge_idx]:
            self._update_k4(k4_idx, old_colour, new_colour)
        
        self.colours[edge_idx] = new_colour
        return 0  # violations tracked internally
    
    def get_violations(self):
        return self.violations
    
    def get_colours(self):
        return self.colours

def _local_search_repair(colours, deadline):
    """Fix violations by changing edge colours using incremental tracking."""
    tracker = _IncrementalTracker(colours)
    current_violations = tracker.get_violations()
    rng = random.Random(42)
    
    # Get list of violating K4s for targeted search
    while time.time() < deadline:
        if current_violations == 0:
            break
        
        # Find a violating K4 and pick a random edge in it
        # Sample random K4s until we find a violating one
        found = False
        for _ in range(20):
            k4_idx = rng.randrange(_NUM_K4S)
            if len(tracker.k4_colour_counts[k4_idx]) < 5:
                found = True
                break
        
        if found:
            # Pick a random edge from the violating K4
            k4_edges = _K4S[k4_idx]
            edge_idx = rng.choice(k4_edges)
        else:
            # Fallback: pick any random edge
            edge_idx = rng.randrange(120)
        
        old_c = tracker.colours[edge_idx]
        # Try colours 0..14
        new_c = rng.randrange(15)
        
        if new_c == old_c:
            continue
        
        # Apply change
        tracker.change_edge(edge_idx, new_c)
        new_violations = tracker.get_violations()
        
        if new_violations < current_violations:
            current_violations = new_violations
        else:
            # Revert
            tracker.change_edge(edge_idx, old_c)
            current_violations = tracker.get_violations()
    
    return tracker.get_colours(), current_violations

def _local_search_minimize(colours, deadline):
    """Minimize number of colours used while maintaining validity."""
    tracker = _IncrementalTracker(colours)
    current_colours = _num_colours_used(colours)
    rng = random.Random(123)
    
    used_colours = list(set(colours))
    
    while time.time() < deadline:
        if len(used_colours) <= 5:
            break
        
        # Try to merge a colour into another
        c_elim = rng.choice(used_colours)
        c_target = rng.choice([c for c in used_colours if c != c_elim])
        
        # Find all edges with c_elim
        edges_to_change = [i for i in range(120) if tracker.colours[i] == c_elim]
        
        if not edges_to_change:
            used_colours = [c for c in used_colours if c != c_elim]
            continue
        
        # Apply all changes
        for e in edges_to_change:
            tracker.change_edge(e, c_target)
        
        v = tracker.get_violations()
        
        if v == 0:
            # Valid, accept
            current_colours = _num_colours_used(tracker.colours)
            used_colours = list(set(tracker.colours))
        else:
            # Revert all changes
            for e in edges_to_change:
                tracker.change_edge(e, c_elim)
            
            # Try random edge swaps to escape local minima
            idx1 = rng.randrange(120)
            idx2 = rng.randrange(120)
            if idx1 == idx2:
                continue
            c1 = tracker.colours[idx1]
            c2 = tracker.colours[idx2]
            if c1 == c2:
                continue
            
            tracker.change_edge(idx1, c2)
            tracker.change_edge(idx2, c1)
            
            v = tracker.get_violations()
            if v == 0:
                current_colours = _num_colours_used(tracker.colours)
                used_colours = list(set(tracker.colours))
            else:
                # Revert
                tracker.change_edge(idx1, c1)
                tracker.change_edge(idx2, c2)

    return tracker.get_colours(), current_colours

def construct():
    t0 = time.time()
    
    # Start with a structured colouring: (i+j) % 15
    # This has good structure and may be easier to optimize
    colours = [((i + j) % 15) for i in range(16) for j in range(i + 1, 16)]
    
    # Phase 1: Repair violations
    deadline_repair = t0 + 40.0
    colours, v = _local_search_repair(colours, deadline_repair)
    
    if v > 0:
        # If we couldn't repair, try random init
        rng = random.Random(0)
        colours = [rng.randrange(15) for _ in range(120)]
        deadline_repair2 = t0 + 60.0
        colours, v = _local_search_repair(colours, deadline_repair2)
    
    if v > 0:
        # Fallback to known valid construction
        colours = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
        v = _count_violations_fast(np.array(colours, dtype=np.int32))
        assert v == 0, "Fallback construction must be valid"
    
    # Phase 2: Minimize colours
    deadline_min = t0 + DEADLINE
    colours, num_c = _local_search_minimize(colours, deadline_min)
    
    # Final verification
    if _count_violations_fast(np.array(colours, dtype=np.int32)) != 0:
        # Fallback
        colours = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    
    return colours
