import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    N = 1000
    max_sum = 2 * N
    
    def can_add(y, S, counts):
        """Check if y can be added to S."""
        if counts[2 * y] >= 2:
            return False
        if len(S) > 0:
            sums = np.array(S) + y
            if np.any(counts[sums] >= 2):
                return False
        return True

    def do_add(y, S, counts, in_set):
        """Add y to S and update counts."""
        if len(S) > 0:
            counts[np.array(S) + y] += 1
        counts[2 * y] += 1
        in_set[y] = True
        S.append(y)

    def do_remove(y, S, counts, in_set):
        """Remove y from S and update counts."""
        if len(S) > 1:
            others = np.array([v for v in S if v != y])
            counts[others + y] -= 1
        counts[2 * y] -= 1
        in_set[y] = False
        S.remove(y)

    def local_search(S, counts, in_set, deadline):
        """Improve set via insertions and swaps."""
        current_size = len(S)
        
        while time.time() < deadline:
            improved = False
            
            # 1. Try to insert any element
            # We can scan all candidates or a random subset.
            # Scanning all is O(N * |S|) which is 1000 * 1400 = 1.4M ops.
            # With vectorization, it's fast.
            candidates = np.where(~in_set)[0]
            candidates = candidates[candidates >= 1]
            
            # Shuffle to avoid bias
            random.shuffle(candidates.tolist())
            
            for y in candidates:
                y = int(y)
                if can_add(y, S, counts):
                    do_add(y, S, counts, in_set)
                    improved = True
                    break
            
            if improved:
                continue
                
            # 2. Try swaps (1-for-1) or 1-for-2
            # Sample some swaps to keep time bounded
            if len(S) < 2:
                break
                
            n_swap_attempts = min(100, len(S)) # Limit attempts per iteration
            
            found_better = False
            for _ in range(n_swap_attempts):
                x_idx = random.randint(0, len(S) - 1)
                x = S[x_idx]
                
                # Pick a random candidate y
                y = int(random.choice(candidates))
                
                # Backup
                S_backup = list(S)
                counts_backup = counts.copy()
                in_set_backup = in_set.copy()
                
                # Remove x
                do_remove(x, S, counts, in_set)
                
                # Try to add y
                if can_add(y, S, counts):
                    do_add(y, S, counts, in_set)
                    
                    # Now try to insert more
                    new_candidates = np.where(~in_set)[0]
                    new_candidates = new_candidates[new_candidates >= 1]
                    random.shuffle(new_candidates.tolist())
                    
                    inserted_extra = False
                    for z in new_candidates:
                        z = int(z)
                        if can_add(z, S, counts):
                            do_add(z, S, counts, in_set)
                            inserted_extra = True
                            break
                            
                    if inserted_extra:
                        # Size increased by 1 (removed 1, added 2)
                        found_better = True
                        break
                    else:
                        # Size same. We might keep this if it helps future moves,
                        # but to be safe and simple, only keep if it allows insertion.
                        # Revert
                        S[:] = S_backup
                        counts[:] = counts_backup
                        in_set[:] = in_set_backup
                else:
                    # Revert
                    S[:] = S_backup
                    counts[:] = counts_backup
                    in_set[:] = in_set_backup
            
            if found_better:
                continue
                
            # 3. If no simple move works, try a "shake":
            # Remove a random element, then greedily add back as many as possible.
            # This is essentially a restart from a perturbed state.
            if time.time() > deadline - 10:
                break
                
            # Simple perturbation: remove 1 random element, then run insert phase again
            # This is covered by the loop if we just continue, but we need to force a change.
            # Let's try removing a random element and NOT adding it back immediately,
            # then the next iteration's insert phase will try to fill the gap.
            
            if len(S) > 0:
                x_idx = random.randint(0, len(S) - 1)
                x = S[x_idx]
                do_remove(x, S, counts, in_set)
                # Do not add it back immediately. The next loop iteration will try to insert.
                # This effectively restarts the local search with a smaller set.
                # To prevent infinite loop of removing the same element, 
                # the insert phase will likely add *some* element.
                # If it adds x back, we are in a loop. 
                # To avoid this, we can track "forbidden" elements temporarily, 
                # or just rely on the randomness of the next insert.
                # Actually, if we remove x, and the next best insert is x, we are stuck.
                # Let's add a small penalty or just accept that we might oscillate.
                # A better way: Remove x, then try to add a *different* element.
                # The insert phase above tries *any* element.
                # If we just remove x and continue, the insert phase will run.
                # If it adds x back, we are in a cycle.
                # To break cycle, we can mark x as "recently removed" and skip it for a few iterations.
                # For simplicity, let's just do the removal and continue.
                # If we oscillate, we waste time.
                # Let's instead try a "2-opt" like move: remove x, add y (where y != x).
                # This is already in the swap phase.
                
                # If swap phase failed, it means no single swap improves size.
                # We can try to remove 2 elements and add 3? Too complex.
                
                # Let's just break if we are stuck for a while.
                # The current loop continues.
                pass
        
        return S, counts, in_set

    best_size = 0
    best_S = []
    
    # Try multiple strategies
    strategies = [
        "asc", "desc", "rand1", "rand2", "rand3"
    ]
    
    for strat in strategies:
        if time.time() > deadline - 20:
            break
            
        counts = np.zeros(max_sum + 1, dtype=np.int32)
        in_set = np.zeros(N + 1, dtype=bool)
        S = []
        
        # Greedy build
        if strat == "asc":
            order = list(range(1, N + 1))
        elif strat == "desc":
            order = list(range(N, 0, -1))
        else:
            order = list(range(1, N + 1))
            random.shuffle(order)
            
        for y in order:
            if in_set[y]:
                continue
            if can_add(y, S, counts):
                do_add(y, S, counts, in_set)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = list(S)
            
        # Local search
        S, counts, in_set = local_search(S, counts, in_set, deadline)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = list(S)
            
    return best_S

if __name__ == "__main__":
    res = construct()
    print(len(res))
