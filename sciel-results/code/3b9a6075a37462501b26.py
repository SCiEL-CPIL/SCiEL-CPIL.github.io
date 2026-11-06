"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin
    
    nums = list(range(2, 257))
    n = len(nums)
    
    # Precompute conflicts
    # conflict[i] is a set of indices j != i that conflict with i
    conflict = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            a, b = nums[i], nums[j]
            g = gcd(a, b)
            l = a * b // g
            if l <= 256:
                conflict[i].add(j)
                conflict[j].add(i)
                
    weights = [1.0 / nums[i] for i in range(n)]
    
    # Greedy Initialization
    # We want to maximize sum of weights.
    # Sort indices by weight descending (smallest nums first)
    # But we must check compatibility.
    
    def get_score(selected_mask):
        return sum(weights[i] for i in range(n) if selected_mask[i])
        
    # Initial Greedy
    selected = [False] * n
    # We pick nodes in order of decreasing weight.
    # If a node is compatible with all currently selected nodes, we pick it.
    # This is a standard greedy for Max Weight Independent Set (approx).
    
    order = sorted(range(n), key=lambda i: -weights[i])
    
    for i in order:
        if not selected[i]:
            # Check if i conflicts with any selected node
            is_compatible = True
            for j in range(n):
                if selected[j] and j in conflict[i]:
                    is_compatible = False
                    break
            if is_compatible:
                selected[i] = True
                
    best_score = get_score(selected)
    best_selected = selected[:]
    
    # Local Search: 1-swap and 2-swap
    # We iterate until no improvement or time limit.
    
    def is_valid(selected_state):
        # Check if selected_state is a valid independent set
        # Optimization: only check pairs where both are selected
        # Since n is small, O(n^2) is fine, but we can optimize using conflict lists
        sel_indices = [i for i in range(n) if selected_state[i]]
        for k, i in enumerate(sel_indices):
            for j in sel_indices[k+1:]:
                if j in conflict[i]:
                    return False
        return True
        
    def try_swaps(selected_state, current_score):
        # 1-swap: remove one selected, add one unselected
        # Or simply: try adding an unselected node if it fits (should have been caught by greedy, 
        # but maybe greedy order prevented it? No, greedy adds if it fits. 
        # However, removing a node might allow adding TWO nodes which sum to more.)
        
        # Let's do 1-exchange: Replace selected i with unselected j
        # This increases score if weight[j] > weight[i] and j is compatible with all selected except i.
        
        sel_indices = [i for i in range(n) if selected_state[i]]
        unsel_indices = [i for i in range(n) if not selected_state[i]]
        
        best_swap = None
        best_gain = 0
        
        for i in sel_indices:
            # If we remove i, which unselected j can we add?
            # j must be compatible with all selected except i.
            # And weight[j] > weight[i] for improvement? Not necessarily, 
            # but we want to maximize.
            # Actually, if weight[j] <= weight[i], swapping doesn't help immediately unless it enables further swaps.
            # For 1-swap, we need w[j] > w[i].
            
            for j in unsel_indices:
                if weights[j] <= weights[i]:
                    continue
                # Check compatibility of j with selected \ {i}
                compatible = True
                for k in sel_indices:
                    if k == i:
                        continue
                    if k in conflict[j]:
                        compatible = False
                        break
                if compatible:
                    # Gain is w[j] - w[i]
                    gain = weights[j] - weights[i]
                    if gain > best_gain:
                        best_gain = gain
                        best_swap = (i, j)
                        
        if best_swap:
            i, j = best_swap
            new_state = selected_state[:]
            new_state[i] = False
            new_state[j] = True
            return new_state, current_score + best_gain
            
        # 2-swap: Replace two selected {i, k} with two unselected {j, l}
        # This is more complex. 
        # Condition: j, l compatible with each other, and with selected \ {i, k}.
        # Gain: (w[j] + w[l]) - (w[i] + w[k])
        
        # To keep it fast, we might skip 2-swap or do a limited version.
        # Given n=255, 2-swap is O(N^4) which is too slow for many iterations.
        # However, we can restrict candidates.
        
        return None, current_score

    improved = True
    while improved and time.time() < deadline:
        improved = False
        new_state, new_score = try_swaps(best_selected, best_score)
        if new_state:
            best_selected = new_state
            best_score = new_score
            improved = True
            
    # Extract result
    result = [nums[i] for i in range(n) if best_selected[i]]
    return result

# EVOLVE-BLOCK-END
