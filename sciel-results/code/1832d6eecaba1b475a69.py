"""
Hypergraph Ramsey R(4,5;3) construction via simulated annealing / local search.

Strategy:
1. Start with a random coloring for a target n (start high, e.g., 20).
2. Use local search to eliminate red K4^(3) and blue K5^(3) violations.
3. Use simulated annealing to escape local minima.
4. If we find a valid coloring for n, try to increase n by adding a vertex
   and re-optimizing.
5. Keep the best valid n found within the time budget.

Key insight: The published construction has 34 points. We aim to find
valid colorings for n as large as possible, ideally reaching 34+.

For efficiency:
- Represent the coloring as a flat array of 0/1 values.
- Precompute which triples belong to each K4 or K5 candidate.
- Use vectorized operations where possible (NumPy).
- Track violations incrementally.
"""

import time
import random
import numpy as np
from itertools import combinations

# Deadline guard: evaluator kills at 150s, we want to return well before.
_DEADLINE = time.time() + 140.0  # 10s safety margin

def _triple_index(a, b, c, n):
    """Map (a,b,c) with 0<=a<b<c<n to index in lexicographic triple list."""
    # Number of triples with first vertex < a: sum_{i=0}^{a-1} C(n-1-i, 2)
    # For simplicity with n up to ~35, we can precompute or use a direct formula.
    # C(n-1-a, 2) etc. Let's use a precomputed table approach via a helper.
    # Actually, for n up to 40, the total triples is C(40,3)=9880, small enough.
    # But we need fast indexing. Let's precompute an index map for given n.
    raise NotImplementedError  # We'll use a precomputed dict instead.

def construct():
    global _DEADLINE
    start_time = time.time()
    
    # We'll try to find the largest n with a valid coloring.
    # Start from a moderate n and grow, or start high and shrink if needed.
    # Better: start at n=20, try to grow.
    
    best_n = 0
    best_colors = None
    
    # Precompute triple lists for various n
    def get_triples(n):
        """Return list of (a,b,c) triples in lexicographic order."""
        return list(combinations(range(n), 3))
    
    def build_index_map(n):
        """Map (a,b,c) -> index in the flat color array."""
        triples = get_triples(n)
        idx_map = {}
        for i, (a, b, c) in enumerate(triples):
            idx_map[(a, b, c)] = i
        return triples, idx_map
    
    def count_red_k4(colors, n, triples, idx_map):
        """Count number of red K4^(3) (4 vertices where all 4 triples are red)."""
        count = 0
        for quad in combinations(range(n), 4):
            all_red = True
            for tri in combinations(quad, 3):
                if colors[idx_map[tri]] != 0:
                    all_red = False
                    break
            if all_red:
                count += 1
        return count
    
    def count_blue_k5(colors, n, triples, idx_map):
        """Count number of blue K5^(3) (5 vertices where all 10 triples are blue)."""
        count = 0
        for quint in combinations(range(n), 5):
            all_blue = True
            for tri in combinations(quint, 3):
                if colors[idx_map[tri]] != 1:
                    all_blue = False
                    break
            if all_blue:
                count += 1
        return count
    
    def total_violations(colors, n, triples, idx_map):
        return count_red_k4(colors, n, triples, idx_map) + count_blue_k5(colors, n, triples, idx_map)
    
    def is_valid(colors, n, triples, idx_map):
        return total_violations(colors, n, triples, idx_map) == 0
    
    # Local search with simulated annealing
    def local_search(n, max_iter=50000, initial_temp=1.0, final_temp=0.01):
        """Try to find a valid coloring for n vertices."""
        if time.time() > _DEADLINE:
            return None, None
        
        triples, idx_map = build_index_map(n)
        num_triples = len(triples)
        
        # Random initial coloring
        colors = np.random.randint(0, 2, size=num_triples).astype(np.int8)
        
        # For efficiency, precompute which K4s contain each triple
        # and which K5s contain each triple
        triple_to_k4s = [[] for _ in range(num_triples)]
        triple_to_k5s = [[] for _ in range(num_triples)]
        
        # Precompute all K4s and K5s
        k4_list = list(combinations(range(n), 4))
        k5_list = list(combinations(range(n), 5))
        
        for k4_idx, quad in enumerate(k4_list):
            for tri in combinations(quad, 3):
                triple_to_k4s[idx_map[tri]].append(k4_idx)
        
        for k5_idx, quint in enumerate(k5_list):
            for tri in combinations(quint, 3):
                triple_to_k5s[idx_map[tri]].append(k5_idx)
        
        # Track violation counts for each K4 and K5
        k4_red_counts = np.zeros(len(k4_list), dtype=np.int32)  # number of red triples in each K4
        k5_blue_counts = np.zeros(len(k5_list), dtype=np.int32)  # number of blue triples in each K5
        
        def recompute_counts():
            nonlocal k4_red_counts, k5_blue_counts
            k4_red_counts[:] = 0
            k5_blue_counts[:] = 0
            for i, (a, b, c) in enumerate(triples):
                if colors[i] == 0:  # red
                    for k4_idx in triple_to_k4s[i]:
                        k4_red_counts[k4_idx] += 1
                else:  # blue
                    for k5_idx in triple_to_k5s[i]:
                        k5_blue_counts[k5_idx] += 1
        
        recompute_counts()
        
        def current_violations():
            return int(np.sum(k4_red_counts == 4)) + int(np.sum(k5_blue_counts == 10))
        
        best_colors = colors.copy()
        best_violations = current_violations()
        temp = initial_temp
        cooling = (final_temp / initial_temp) ** (1.0 / max_iter)
        
        for it in range(max_iter):
            if time.time() > _DEADLINE:
                break
            
            # Pick a random triple to flip
            tri_idx = random.randint(0, num_triples - 1)
            
            old_color = colors[tri_idx]
            new_color = 1 - old_color
            
            # Compute delta in violations
            delta = 0
            
            # Effect on K4s containing this triple
            for k4_idx in triple_to_k4s[tri_idx]:
                if old_color == 0:  # was red, now blue
                    # red count decreases by 1
                    if k4_red_counts[k4_idx] == 4:
                        delta -= 1  # remove a red K4
                    if k4_red_counts[k4_idx] - 1 == 4:
                        delta += 1  # create a red K4 (impossible since it was 4)
                    # Actually: if it was 4 and we remove one red, it's no longer all-red.
                    # If it was 3 and we add one red (flipping blue->red), it becomes 4.
                else:  # was blue, now red
                    if k4_red_counts[k4_idx] == 3:
                        delta += 1  # create a red K4
                    if k4_red_counts[k4_idx] == 4:
                        delta -= 1  # remove a red K4 (impossible since it was 4)
            
            # Effect on K5s containing this triple
            for k5_idx in triple_to_k5s[tri_idx]:
                if old_color == 1:  # was blue, now red
                    if k5_blue_counts[k5_idx] == 10:
                        delta -= 1  # remove a blue K5
                    if k5_blue_counts[k5_idx] - 1 == 10:
                        delta += 1  # impossible
                else:  # was red, now blue
                    if k5_blue_counts[k5_idx] == 9:
                        delta += 1  # create a blue K5
                    if k5_blue_counts[k5_idx] == 10:
                        delta -= 1  # impossible
            
            new_violations = best_violations + delta
            
            # Accept or reject
            if new_violations < best_violations or random.random() < np.exp(-delta / max(temp, 1e-10)):
                # Apply the flip
                colors[tri_idx] = new_color
                for k4_idx in triple_to_k4s[tri_idx]:
                    if old_color == 0:
                        k4_red_counts[k4_idx] -= 1
                    else:
                        k4_red_counts[k4_idx] += 1
                for k5_idx in triple_to_k5s[tri_idx]:
                    if old_color == 1:
                        k5_blue_counts[k5_idx] -= 1
                    else:
                        k5_blue_counts[k5_idx] += 1
                
                if new_violations < best_violations:
                    best_violations = new_violations
                    best_colors = colors.copy()
                    if best_violations == 0:
                        return best_colors, (n, triples, idx_map)
            else:
                # Reject, keep old state (no changes needed since we didn't modify)
                pass
            
            temp *= cooling
        
        return best_colors, (n, triples, idx_map)
    
    # Try increasing n from 15 up to 35
    for n in range(15, 36):
        if time.time() > _DEADLINE:
            break
        
        # Try a few restarts for each n
        found = False
        for restart in range(3):
            if time.time() > _DEADLINE:
                break
            
            colors, info = local_search(n, max_iter=30000, initial_temp=1.0, final_temp=0.01)
            if colors is not None:
                n_val, triples, idx_map = info
                if is_valid(colors, n_val, triples, idx_map):
                    if n_val > best_n:
                        best_n = n_val
                        best_colors = colors.tolist()
                    found = True
                    break
        
        if found:
            # We found a valid coloring for n, try to grow
            continue
        else:
            # Couldn't find valid for n, try n+1 might be easier? No, harder.
            # Break and return best found.
            # Actually, let's try a couple more n values in case of bad luck.
            continue
    
    if best_colors is not None:
        return {'n': best_n, 'colors': best_colors}
    
    # Fallback: return the seed solution
    from itertools import combinations as comb
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in comb(range(n), 3)]}
