import random
import math
from itertools import combinations

def construct():
    # 1. Generate all 4-subsets as bitmasks
    all_sets = []
    for c in combinations(range(1, 17), 4):
        mask = 0
        for x in c:
            mask |= (1 << (x - 1))
        all_sets.append(mask)
    
    N = len(all_sets)
    
    # 2. Precompute intersection matrix
    # inter[i][j] is the bitmask of intersection of all_sets[i] and all_sets[j]
    inter = [[0]*N for _ in range(N)]
    for i in range(N):
        for j in range(i+1, N):
            k = all_sets[i] & all_sets[j]
            inter[i][j] = k
            inter[j][i] = k
        inter[i][i] = all_sets[i]

    # Helper to count sunflowers involving a specific set S_idx with a list of selected indices
    def count_sunflowers_with_S(S_idx, selected_indices):
        count = 0
        map_K = {}
        for A_idx in selected_indices:
            K = inter[A_idx][S_idx]
            if K in map_K:
                for B_idx in map_K[K]:
                    if inter[A_idx][B_idx] == K:
                        count += 1
            else:
                map_K[K] = []
            map_K[K].append(A_idx)
        return count

    # 3. Initial Solution using Greedy (Random Order)
    # We do this to start with a valid, reasonably large set
    def greedy_init():
        indices = list(range(N))
        random.shuffle(indices)
        selected = []
        for idx in indices:
            # Check if idx can be added to selected
            # We just use the count function, if 0, it's valid
            if count_sunflowers_with_S(idx, selected) == 0:
                selected.append(idx)
        return selected

    best_selected = greedy_init()
    best_size = len(best_selected)
    
    # 4. Simulated Annealing
    T = 1.0
    T_min = 0.01
    cooling = 0.995
    M = 100.0 # Penalty for sunflowers
    
    current_selected = best_selected[:]
    current_num_sunflowers = 0 # Valid state
    current_size = len(current_selected)
    
    # To allow fast removal, we might need to track which sunflowers are present?
    # No, we just recalculate delta.
    # But removing a set S reduces sunflowers by the number of sunflowers involving S.
    # We can compute this on the fly.
    
    for step in range(50000): # Max steps
        if T < T_min:
            break
            
        # Choose a move
        move_type = random.choice(['add', 'remove', 'swap'])
        
        new_selected = current_selected[:]
        delta_size = 0
        delta_sunflowers = 0
        
        if move_type == 'add':
            # Pick random unselected
            unselected = [i for i in range(N) if i not in current_selected]
            if not unselected:
                continue
            idx = random.choice(unselected)
            new_selected.append(idx)
            delta_size = 1
            # Count new sunflowers
            delta_sunflowers = count_sunflowers_with_S(idx, current_selected)
            
        elif move_type == 'remove':
            if not current_selected:
                continue
            idx = random.choice(current_selected)
            new_selected.remove(idx)
            delta_size = -1
            # Count sunflowers removed (those involving idx)
            delta_sunflowers = -count_sunflowers_with_S(idx, [i for i in current_selected if i != idx])
            
        else: # swap
            if not current_selected:
                continue
            unselected = [i for i in range(N) if i not in current_selected]
            if not unselected:
                continue
            idx_out = random.choice(current_selected)
            idx_in = random.choice(unselected)
            
            # Remove idx_out, Add idx_in
            # Sunflowers involving idx_out are removed
            # Sunflowers involving idx_in are added
            # But we must be careful: sunflowers involving both? No, they are distinct.
            # However, the set of selected changes.
            # Let's just compute the total sunflowers in the new state?
            # No, that's expensive.
            # Delta = Sunflowers_in_new - Sunflowers_in_old
            # Sunflowers_in_old = current_num_sunflowers (which is 0 if we maintain validity, but SA allows invalid)
            # If we allow invalid states, we need to track current_num_sunflowers.
            
            # Let's stick to valid states only? 
            # If we only accept valid states, it's just Hill Climbing.
            # To explore, we need to allow invalid states.
            
            # If we allow invalid states, we must maintain `current_num_sunflowers`.
            # For swap:
            # Remove idx_out: subtract sunflowers involving idx_out (with other current selected)
            # Add idx_in: add sunflowers involving idx_in (with new selected, which is current - idx_out + idx_in)
            # This is tricky because the "other selected" set changes.
            
            # Simpler: Just compute the number of sunflowers in `new_selected`?
            # No, too slow.
            
            # Alternative: Only use Add and Remove moves.
            # This allows moving up and down in size.
            # It can escape local optima by removing a set to allow adding two others?
            # Remove 1 (size -1, sunflowers -k), Add 1 (size +1, sunflowers +m).
            # Net size 0.
            # To increase size, we need Add moves that don't increase sunflowers (or decrease).
            
            # Let's just use Add and Remove.
            pass

        # Calculate Energy Change
        # Energy = -Size + M * Sunflowers
        # We want to minimize Energy.
        # Delta E = -delta_size + M * delta_sunflowers
        
        delta_E = -delta_size + M * delta_sunflowers
        
        # Acceptance
        if delta_E < 0:
            accept = True
        else:
            accept = random.random() < math.exp(-delta_E / T)
            
        if accept:
            current_selected = new_selected
            current_num_sunflowers += delta_sunflowers
            current_size += delta_size
            
            if current_num_sunflowers == 0 and current_size > best_size:
                best_size = current_size
                best_selected = current_selected[:]
                
        T *= cooling

    # Convert best_selected indices to lists
    result = []
    for idx in best_selected:
        mask = all_sets[idx]
        s = []
        for i in range(16):
            if mask & (1 << i):
                s.append(i + 1)
        result.append(sorted(s))
        
    return result
