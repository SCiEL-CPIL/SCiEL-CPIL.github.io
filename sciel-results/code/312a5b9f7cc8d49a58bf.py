import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin for the 90s limit
    
    N = 512
    
    # Precompute all 4-APs
    # We store for each element x, the list of tuples (other1, other2, other3)
    # representing the other three elements in the AP that contains x.
    # This allows O(1) lookup for conflicts when checking if x can be added.
    
    ap_list = [[] for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        # d must be such that a + 3d <= N
        max_d = (N - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            
            # AP: (a, b, c, e)
            # If we add a, conflict if b, c, e are in S
            ap_list[a].append((b, c, e))
            # If we add b, conflict if a, c, e are in S
            ap_list[b].append((a, c, e))
            # If we add c, conflict if a, b, e are in S
            ap_list[c].append((a, b, e))
            # If we add e, conflict if a, b, c are in S
            ap_list[e].append((a, b, c))

    def can_add(S_bits, x):
        """Check if x can be added to S (represented as bits) without forming a 4-AP."""
        idx = x - 1
        if (S_bits >> idx) & 1:
            return False  # Already in S
        
        for (o1, o2, o3) in ap_list[x]:
            # Check if o1, o2, o3 are all in S
            if ((S_bits >> (o1 - 1)) & 1) and ((S_bits >> (o2 - 1)) & 1) and ((S_bits >> (o3 - 1)) & 1):
                return False
        return True

    def get_elements(S_bits):
        """Return list of elements in S."""
        return [i + 1 for i in range(N) if (S_bits >> i) & 1]

    # --- Phase 1: Greedy Construction ---
    # We run multiple greedy trials with different random orderings to find a good starting point.
    # The current program's approach is solid here.
    
    best_S_bits = 0
    best_count = 0
    
    # Run greedy trials
    # We can afford ~50-100 trials quickly.
    num_greedy_trials = 100
    
    for i in range(num_greedy_trials):
        if time.time() > deadline - 10.0:  # Leave 10s for SA
            break
        
        random.seed(i)
        S_bits = 0
        count = 0
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if can_add(S_bits, x):
                S_bits |= (1 << (x - 1))
                count += 1
        
        if count > best_count:
            best_count = count
            best_S_bits = S_bits

    # --- Phase 2: Simulated Annealing / Local Search ---
    # We improve upon the current program's SA by:
    # 1. Using a more robust move set (Swap, Add, Remove).
    # 2. Implementing a "fill pass" periodically to ensure we don't miss easy additions.
    # 3. Better cooling schedule.
    # 4. Tracking the best solution found.

    S_bits = best_S_bits
    count = best_count
    S_list = get_elements(S_bits)
    
    # We maintain a list of elements NOT in S for efficient random selection of additions
    # But maintaining this list is O(N) to update. 
    # Since N=512, random selection from 1..N and checking membership is O(1) amortized.
    # Let's stick to random selection from 1..N for simplicity and speed.

    all_elements = list(range(1, N + 1))
    
    # SA Parameters
    T = 10.0
    cooling = 0.9995
    min_T = 0.1
    
    # We will run SA until time runs out.
    # To escape local optima, we occasionally do a "hard" restart or a more aggressive move.
    # But the current program's swap/add is decent. Let's refine the acceptance criteria.
    
    # Improvement: 
    # Instead of just swapping, we can try to "repair" a conflict.
    # If we can't add x, we can remove one of the conflicting elements y and try to add x.
    # This is effectively a swap, but guided by the conflict.
    
    # Let's implement a guided local search:
    # 1. Try to add random x.
    # 2. If fails, pick a random conflicting element y from the AP that blocked x.
    # 3. Remove y, try to add x.
    # 4. If successful, we've swapped y for x (size unchanged, but structure changed).
    # 5. Also, try simple random swaps.
    # 6. Also, try random additions (if size < N).
    
    # To maximize cardinality, we primarily want to ADD.
    # If we can't add, we want to SWAP to create space for a future add.
    
    # Let's use a temperature-based approach where we allow temporary size decreases (removals)
    # if it leads to a better configuration later? No, that's risky.
    # Better: Only accept moves that maintain or increase size.
    # - Add: size +1 (Always accept)
    # - Swap: size 0 (Accept with prob 1, or use SA to explore)
    # - Remove: size -1 (Rarely accept, only if T is high)
    
    # Actually, the "Inspiration 1" uses a Hitting Set removal then add-back.
    # The "Inspiration 2" uses Greedy + Local Search (Swaps).
    # The current program uses Greedy + SA (Swaps/Add).
    
    # Let's combine:
    # 1. Start with best Greedy.
    # 2. Run a local search that focuses on "guided swaps" (remove blocker, add target).
    # 3. Periodically run a "fill pass" (try to add every possible element).
    
    last_fill_time = time.time()
    fill_interval = 1.0  # seconds
    
    while time.time() < deadline - 2.0:
        # Check if it's time for a fill pass
        if time.time() - last_fill_time > fill_interval:
            # Fill pass: try to add any element that can be added
            changed = True
            while changed:
                changed = False
                for x in all_elements:
                    if can_add(S_bits, x):
                        S_bits |= (1 << (x - 1))
                        S_list.append(x)
                        count += 1
                        changed = True
                        if count > best_count:
                            best_count = count
                            best_S_bits = S_bits
            last_fill_time = time.time()
        
        # Choose a move
        move_type = random.random()
        
        if move_type < 0.6:
            # Guided Swap: Pick random x not in S. If it can't be added, remove a blocker.
            x = random.choice(all_elements)
            if (S_bits >> (x - 1)) & 1:
                continue
            
            # Check if x can be added
            if can_add(S_bits, x):
                # Direct add!
                S_bits |= (1 << (x - 1))
                S_list.append(x)
                count += 1
                if count > best_count:
                    best_count = count
                    best_S_bits = S_bits
                continue
            
            # x cannot be added. Find a conflicting AP.
            # We need to find one AP where the other 3 are in S.
            conflicting_y = None
            for (o1, o2, o3) in ap_list[x]:
                if ((S_bits >> (o1 - 1)) & 1) and ((S_bits >> (o2 - 1)) & 1) and ((S_bits >> (o3 - 1)) & 1):
                    # Pick one of o1, o2, o3 to remove
                    # Heuristic: Remove the one with the most conflicts? Or random?
                    # Random is fine for now.
                    conflicting_y = random.choice([o1, o2, o3])
                    break
            
            if conflicting_y is None:
                continue
            
            # Try to remove conflicting_y and add x
            # Remove conflicting_y
            S_bits ^= (1 << (conflicting_y - 1))
            if conflicting_y in S_list:
                S_list.remove(conflicting_y)
            count -= 1
            
            # Try to add x
            if can_add(S_bits, x):
                S_bits |= (1 << (x - 1))
                S_list.append(x)
                count += 1
                # Swap successful. Size unchanged.
                # We don't update best_count unless we found a better one (we didn't, size same).
                # But the structure is different, which might allow future adds.
            else:
                # Failed to add x. Revert removal.
                S_bits |= (1 << (conflicting_y - 1))
                S_list.append(conflicting_y)
                count += 1
        
        elif move_type < 0.9:
            # Random Swap: Remove random y from S, Add random x not in S.
            if not S_list:
                continue
            y = random.choice(S_list)
            x = random.choice(all_elements)
            if x == y:
                continue
            
            # Remove y
            S_bits ^= (1 << (y - 1))
            S_list.remove(y)
            count -= 1
            
            # Try to add x
            if can_add(S_bits, x):
                S_bits |= (1 << (x - 1))
                S_list.append(x)
                count += 1
            else:
                # Revert
                S_bits |= (1 << (y - 1))
                S_list.append(y)
                count += 1
        
        else:
            # Random Add: Try to add a random x.
            x = random.choice(all_elements)
            if can_add(S_bits, x):
                S_bits |= (1 << (x - 1))
                S_list.append(x)
                count += 1
                if count > best_count:
                    best_count = count
                    best_S_bits = S_bits
        
        # Cooling (if we were using SA properly, but here we are mostly greedy/guided)
        # We can keep T constant or decrease it slowly to encourage stability?
        # Actually, since we don't accept worse moves (size decrease), T is not strictly needed for acceptance.
        # But we can use it to decide how "aggressive" we are with removals?
        # Let's keep it simple.
        
        # Update best
        if count > best_count:
            best_count = count
            best_S_bits = S_bits

    # Final Fill Pass
    for x in all_elements:
        if time.time() > deadline - 1.0:
            break
        if can_add(best_S_bits, x):
            best_S_bits |= (1 << (x - 1))
            best_count += 1

    result = [i + 1 for i in range(N) if (best_S_bits >> i) & 1]
    return sorted(result)
