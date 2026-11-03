import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    total_words = q ** n
    
    # 1. Precompute neighbors (Hamming distance <= radius)
    # For q=3, n=6, radius=1: neighbors are the word itself + 2*6 = 13 words
    neigh = [[] for _ in range(total_words)]
    neigh_set = [set() for _ in range(total_words)]
    
    for idx in range(total_words):
        neigh[idx].append(idx)
        neigh_set[idx].add(idx)
        # Iterate through positions
        temp = idx
        for i in range(n):
            pow_q_i = q ** i
            digit = (idx // pow_q_i) % q
            for new_digit in range(q):
                if new_digit != digit:
                    new_idx = idx + (new_digit - digit) * pow_q_i
                    neigh[idx].append(new_idx)
                    neigh_set[idx].add(new_idx)
    
    # Convert neigh to numpy arrays for faster coverage updates
    neigh_np = [np.array(neigh[i], dtype=np.int64) for i in range(total_words)]
    
    # 2. Build Distance-2 Graph for Independent Set Initialization
    # Two words are connected if their Hamming distance is exactly 2.
    # A maximal independent set in this graph gives a set of words where no two are at distance 2.
    dist2_adj = [set() for _ in range(total_words)]
    
    # Optimization: Only build the graph if we have time. 
    # For 729 nodes, this is fast.
    for idx in range(total_words):
        for i in range(n):
            pow_q_i = q ** i
            digit_i = (idx // pow_q_i) % q
            for j in range(i + 1, n):
                pow_q_j = q ** j
                digit_j = (idx // pow_q_j) % q
                # Change digit i and j
                for new_i in range(q):
                    if new_i == digit_i:
                        continue
                    for new_j in range(q):
                        if new_j == digit_j:
                            continue
                        new_idx = idx + (new_i - digit_i) * pow_q_i + (new_j - digit_j) * pow_q_j
                        if new_idx != idx:
                            dist2_adj[idx].add(new_idx)
                            dist2_adj[new_idx].add(idx)
    
    rng = random.Random(42)
    
    # 3. Greedy Independent Set Initialization
    # We want a maximal independent set in the distance-2 graph.
    # Strategy: Randomly shuffle vertices, then greedily pick a vertex if it doesn't conflict with already picked vertices.
    
    candidates = list(range(total_words))
    rng.shuffle(candidates)
    
    code_set = set()
    in_set = [False] * total_words
    
    # Greedy Independent Set
    for v in candidates:
        if in_set[v]:
            continue
        # Check if v conflicts with any node in code_set
        # Conflict if u in code_set and u is in dist2_adj[v]
        is_conflict = False
        for u in dist2_adj[v]:
            if in_set[u]:
                is_conflict = True
                break
        if not is_conflict:
            code_set.add(v)
            in_set[v] = True
    
    # 4. Repair to Cover
    # The independent set might not cover the whole space.
    # We need to add words until it's a valid cover.
    counts = np.zeros(total_words, dtype=np.int32)
    for c in code_set:
        counts[neigh_np[c]] += 1
            
    # Find uncovered words
    uncovered_mask = (counts == 0)
    uncovered_indices = np.where(uncovered_mask)[0]
    
    # Greedily add words to cover uncovered space
    # We prioritize words that cover the most uncovered nodes.
    while len(uncovered_indices) > 0 and time.time() < deadline - 5.0:
        best_c = -1
        best_gain = 0
        
        # Sample some candidates to speed up? Or iterate all? 729 is small.
        for c in range(total_words):
            if c in code_set:
                continue
            # Calculate gain: number of currently uncovered nodes covered by c
            # c covers neigh[c]. We want the intersection of neigh[c] and uncovered_indices.
            # Since uncovered_indices is an array, we can use numpy operations if we convert neigh[c] to set/array
            # But simple loop is fine for 13 neighbors.
            gain = 0
            for u in neigh[c]:
                if counts[u] == 0:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_c = c
        
        if best_c == -1:
            # Should not happen if we iterate all, but safety break
            break
            
        code_set.add(best_c)
        counts[neigh_np[best_c]] += 1
        
        # Recalculate uncovered for next iteration (cheap for 729)
        uncovered_mask = (counts == 0)
        uncovered_indices = np.where(uncovered_mask)[0]
        
    # If still uncovered (shouldn't happen with greedy), force add
    if len(uncovered_indices) > 0:
        for u in uncovered_indices:
            u_int = int(u)
            if u_int not in code_set:
                code_set.add(u_int)
                counts[neigh_np[u_int]] += 1
                # Update uncovered mask locally or just assume fixed? 
                # Actually, adding u covers u and its neighbors. 
                # It's safer to just recalculate at the end or assume the greedy loop handled it.
                # Given the density, the greedy loop should have finished.
    
    best_code = set(code_set)
    best_size = len(code_set)
    
    # 5. Simulated Annealing
    # Moves:
    # 1. Remove: If a codeword is redundant (all its neighbors are covered by others), remove it.
    # 2. Swap: Replace a codeword with another if the new one covers all unique neighbors of the old one.
    
    temp = 1.0
    cool_rate = 0.9995
    min_temp = 0.01
    
    def is_removable(c):
        for u in neigh[c]:
            if counts[u] == 1:
                return False
        return True

    def is_valid_swap(c, new_c):
        # Check if removing c leaves any neighbor uncovered by new_c
        for u in neigh[c]:
            if counts[u] == 1 and u not in neigh_set[new_c]:
                return False
        return True

    while time.time() < deadline:
        if temp < min_temp:
            temp = min_temp
            
        move_type = rng.random()
        
        if move_type < 0.5:
            # Try Remove
            if len(code_set) > 0:
                c = rng.choice(list(code_set))
                if is_removable(c):
                    # Perform removal
                    code_set.discard(c)
                    counts[neigh_np[c]] -= 1
                    if len(code_set) < best_size:
                        best_size = len(code_set)
                        best_code = set(code_set)
                        
        elif move_type < 0.9:
            # Try Swap
            if len(code_set) > 0:
                c = rng.choice(list(code_set))
                new_c = rng.randint(0, total_words - 1)
                
                if new_c == c or new_c in code_set:
                    continue
                    
                if is_valid_swap(c, new_c):
                    # Perform swap
                    code_set.discard(c)
                    code_set.add(new_c)
                    counts[neigh_np[c]] -= 1
                    counts[neigh_np[new_c]] += 1
                    if len(code_set) < best_size:
                        best_size = len(code_set)
                        best_code = set(code_set)
                else:
                    # If invalid, we can still accept it with some probability in SA?
                    # Usually for covering codes, we maintain validity (hard constraint).
                    # So we only accept valid moves.
                    pass
        else:
            # Random restart or large jump?
            # Try adding a random word (increases size, might allow better structure later)
            # Or try a "kick": remove a random word and add a random replacement even if invalid?
            # No, invalid is bad.
            # Let's try a "shuffle" of a random subset? Too complex.
            # Just stick to Remove/Swap.
            pass
            
        temp *= cool_rate
        
    # Restore best code
    code_set = best_code
    
    # Final cleanup: remove any remaining redundancies
    # We need to recompute counts for the best code if we restored it?
    # Yes, counts might be out of sync if we restored from a previous state.
    # Actually, in the loop above, we updated counts incrementally.
    # But `best_code` is a snapshot. `code_set` at the end of the loop is the current state.
    # We should check if the current state is better than best_code?
    # The loop tracks best_size.
    # Let's ensure we are working with the best code found.
    
    # Recompute counts for the final best_code to be safe for the final cleanup pass
    counts = np.zeros(total_words, dtype=np.int32)
    for c in best_code:
        counts[neigh_np[c]] += 1
        
    # Greedy remove pass on the best code
    changed = True
    while changed and time.time() < deadline - 1:
        changed = False
        # Find removable
        removable = []
        for c in best_code:
            if is_removable(c):
                removable.append(c)
        if removable:
            c = rng.choice(removable)
            best_code.discard(c)
            counts[neigh_np[c]] -= 1
            changed = True
            if len(best_code) < best_size:
                best_size = len(best_code)

    # Convert back to list of lists
    def int_to_list(val, n):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = val % q
            val //= q
        return word

    result = [int_to_list(c, n) for c in best_code]
    return result
