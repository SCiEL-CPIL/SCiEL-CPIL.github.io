"""Optimized solver for large reciprocal weight under LCM separation."""
# EVOLVE-BLOCK-START
import math
from math import gcd
import random

def construct():
    """
    Find a nonempty subset of [2, 256] such that lcm(a,b) > 256 for all distinct pairs,
    maximizing the sum of reciprocals.
    """
    MAX_N = 256
    N = MAX_N - 1  # Number of candidates: 2..256
    
    # Map value to index 0..254
    # val = index + 2
    
    # Precompute incompatibility matrix
    # incompat[i][j] is True if lcm(val_i, val_j) <= 256
    # We use a list of sets for memory efficiency and fast lookup
    incompat = [set() for _ in range(N)]
    for i in range(N):
        vi = i + 2
        for j in range(i + 1, N):
            vj = j + 2
            g = gcd(vi, vj)
            if vi // g * vj <= 256:
                incompat[i].add(j)
                incompat[j].add(i)
                
    # Objective function
    def get_score(selected_indices):
        return sum(1.0 / (i + 2) for i in selected_indices)
        
    # Check if a node can be added to the selected set
    def can_add(node, selected_set):
        # Check if node is incompatible with any in selected_set
        for s in selected_set:
            if s in incompat[node]:
                return False
        return True

    # Greedy Construction
    def greedy_construct(order):
        selected = []
        selected_set = set()
        for node in order:
            if can_add(node, selected_set):
                selected.append(node)
                selected_set.add(node)
        return selected, selected_set

    # Local Search
    def local_search(selected, selected_set):
        improved = True
        while improved:
            improved = False
            # Try adding
            for node in range(N):
                if node not in selected_set and can_add(node, selected_set):
                    selected.append(node)
                    selected_set.add(node)
                    improved = True
                    # We don't break, we keep adding if possible, but 
                    # adding one might block others? No, adding a node only 
                    # adds constraints. If node A is compatible with current set,
                    # and node B is compatible with current set, A and B might 
                    # be incompatible with each other. 
                    # So we should check against the *new* set.
                    # The loop above re-iterates? No, it's a single pass.
                    # Better to restart the while loop or use a queue.
                    # Let's break and restart the while loop to be safe and simple.
                    break
                    
            if improved:
                continue
            
            # Try swapping
            best_gain = 0
            best_swap = None
            for i, s in enumerate(selected):
                # Try replacing s with c
                # c must be compatible with all selected except s
                others = [x for j, x in enumerate(selected) if j != i]
                others_set = selected_set - {s}
                
                for c in range(N):
                    if c in selected_set:
                        continue
                    # Check compatibility with others
                    if any(c in incompat[o] for o in others_set):
                        continue
                    # Check gain
                    gain = 1.0 / (c + 2) - 1.0 / (s + 2)
                    if gain > best_gain:
                        best_gain = gain
                        best_swap = (s, c)
                        
            if best_swap:
                s, c = best_swap
                selected.remove(s)
                selected_set.remove(s)
                selected.append(c)
                selected_set.add(c)
                improved = True
            else:
                # Try removing? 
                # Removing a node always decreases score, but might allow adding 
                # multiple nodes later? 
                # The swap step handles 1-for-1. 
                # If we remove s, we might add c1, c2. 
                # This is a 1-for-k move. 
                # Given the reciprocal nature, 1-for-k is rare to be beneficial 
                # unless s is very large (small reciprocal) and c1, c2 are small.
                # But if c1, c2 are small, they likely conflict with each other 
                # or with the rest of the set.
                # We'll stick to 1-for-1 swaps and adds for now, which is standard 
                # for this size.
                pass
                
        return selected, selected_set

    # Generate initial solutions
    candidates = []
    
    # 1. Greedy Ascending (2, 3, 4...)
    order_asc = list(range(N))
    sel1, set1 = greedy_construct(order_asc)
    sel1, set1 = local_search(sel1, set1)
    candidates.append((get_score(sel1), sel1, set1))
    
    # 2. Greedy Descending (256, 255...)
    order_desc = list(range(N-1, -1, -1))
    sel2, set2 = greedy_construct(order_desc)
    sel2, set2 = local_search(sel2, set2)
    candidates.append((get_score(sel2), sel2, set2))
    
    # 3. Greedy by Reciprocal (which is same as Ascending value)
    # 4. Random Restarts
    random.seed(42)
    for _ in range(50):
        order_rand = list(range(N))
        random.shuffle(order_rand)
        sel_r, set_r = greedy_construct(order_rand)
        sel_r, set_r = local_search(sel_r, set_r)
        candidates.append((get_score(sel_r), sel_r, set_r))

    # Select best
    best_score, best_sel, best_set = max(candidates, key=lambda x: x[0])
    
    # Convert indices back to values
    result = [i + 2 for i in best_sel]
    
    return result

# EVOLVE-BLOCK-END
