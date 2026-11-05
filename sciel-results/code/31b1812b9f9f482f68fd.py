import time
import random

N = 256
# 1-indexed for problem statement, 0-indexed for bitmasks
# Bit i corresponds to number i+1

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Precomputation
    # 4-AP conflicts
    # conflict_4ap[x] is a list of tuples (u, v, w) such that {x, u, v, w} is a 4-AP
    # We use 0-indexed indices internally.
    conflict_4ap = [[] for _ in range(N)]
    
    # 3-AP gains
    # gain_middle[x]: list of (u, v) pairs for (u, x, v)
    # gain_first[x]: list of (u, v) pairs for (x, u, v)
    # gain_last[x]: list of (u, v) pairs for (u, v, x)
    gain_middle = [[] for _ in range(N)]
    gain_first = [[] for _ in range(N)]
    gain_last = [[] for _ in range(N)]
    
    # Generate 4-APs
    for a in range(N):
        for d in range(1, (N - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e >= N:
                break
            # 4-AP: a, b, c, e
            conflict_4ap[a].append((b, c, e))
            conflict_4ap[b].append((a, c, e))
            conflict_4ap[c].append((a, b, e))
            conflict_4ap[e].append((a, b, c))
            
    # Generate 3-AP gain pairs
    for a in range(N):
        for d in range(1, (N - a) // 2 + 1):
            b = a + d
            c = a + 2 * d
            if c >= N:
                break
            # 3-AP: a, b, c
            # If adding b: need a, c. gain_middle[b].append((a, c))
            gain_middle[b].append((a, c))
            # If adding a: need b, c. gain_first[a].append((b, c))
            gain_first[a].append((b, c))
            # If adding c: need a, b. gain_last[c].append((a, b))
            gain_last[c].append((a, b))

    def is_valid_add(mask, x):
        """Check if adding x to mask creates a 4-AP."""
        for (u, v, w) in conflict_4ap[x]:
            if (mask >> u) & 1 and (mask >> v) & 1 and (mask >> w) & 1:
                return False
        return True

    def get_gain(mask, x):
        """Calculate number of new 3-APs formed by adding x to mask."""
        gain = 0
        for (u, v) in gain_middle[x]:
            if (mask >> u) & 1 and (mask >> v) & 1:
                gain += 1
        for (u, v) in gain_first[x]:
            if (mask >> u) & 1 and (mask >> v) & 1:
                gain += 1
        for (u, v) in gain_last[x]:
            if (mask >> u) & 1 and (mask >> v) & 1:
                gain += 1
        return gain

    def get_loss(mask, x):
        """Calculate number of 3-APs destroyed by removing x from mask."""
        # Same as gain, but we are removing x.
        # The 3-APs containing x are exactly the ones that would be formed by adding x to mask \ {x}.
        # So we calculate gain on mask with x removed.
        mask_without_x = mask & ~(1 << x)
        return get_gain(mask_without_x, x)

    def count_3aps(mask):
        """Total count of 3-APs. Used for initialization and verification."""
        total = 0
        # Sum gains for all x in mask? No, that double counts?
        # Each 3-AP (a, b, c) is counted once when adding c (if a,b present), once when adding b (if a,c present - wait, b is middle), once when adding a (if b,c present).
        # Actually, get_gain(mask, x) counts 3-APs where x is the *new* element.
        # If we build the set incrementally, sum of gains is the total.
        # But for an existing mask, we can just count directly.
        # Direct count:
        for a in range(N):
            if not ((mask >> a) & 1):
                continue
            for d in range(1, (N - a) // 2 + 1):
                b = a + d
                c = a + 2 * d
                if c >= N:
                    break
                if ((mask >> b) & 1) and ((mask >> c) & 1):
                    total += 1
        return total

    def local_search(initial_mask, time_limit):
        """
        Local search starting from initial_mask.
        Returns best_mask found.
        """
        mask = initial_mask
        current_score = count_3aps(mask)
        best_mask = mask
        best_score = current_score
        
        # Precompute list of elements for faster iteration
        elements_in = [i for i in range(N) if (mask >> i) & 1]
        elements_out = [i for i in range(N) if not ((mask >> i) & 1)]
        
        while time.time() < time_limit:
            improved = False
            
            # 1. Try Additions
            best_gain = 0
            best_x_add = -1
            for x in elements_out:
                if not is_valid_add(mask, x):
                    continue
                g = get_gain(mask, x)
                if g > best_gain:
                    best_gain = g
                    best_x_add = x
            
            if best_x_add != -1 and best_gain > 0:
                mask |= (1 << best_x_add)
                elements_out.remove(best_x_add)
                elements_in.append(best_x_add)
                current_score += best_gain
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_mask = mask
                continue # Restart loop to re-evaluate all options

            # 2. Try Swaps (Remove one, Add one)
            # To be efficient, we iterate over candidates to ADD, and check if we can remove a blocker.
            # Or iterate over elements to REMOVE, and find best ADD.
            # Let's try: For each x in elements_in, remove it, then find best add y.
            # This is O(|In| * |Out| * Cost). Might be slow.
            # Alternative: For each y in elements_out, if it's blocked, find a blocker x to remove.
            
            best_swap_gain = 0
            best_x_remove = -1
            best_y_add = -1
            
            # Heuristic: Only try swaps if we are stuck on additions
            # We limit the number of swap checks per iteration to avoid timeout
            swap_check_limit = 200 
            checks = 0
            
            for y in elements_out:
                if checks >= swap_check_limit:
                    break
                # Check if y is valid
                if is_valid_add(mask, y):
                    # This should have been caught by Addition phase if gain > 0
                    # If gain was 0, it's not useful.
                    continue
                
                # Find blockers
                blockers = []
                for (u, v, w) in conflict_4ap[y]:
                    if (mask >> u) & 1 and (mask >> v) & 1 and (mask >> w) & 1:
                        # u, v, w are in mask. Any of them can be a blocker if we remove one of them?
                        # No, removing one of u,v,w breaks the 4-AP {y, u, v, w}.
                        # So any of u, v, w is a valid "removal candidate" to make y valid.
                        blockers.extend([u, v, w])
                
                if not blockers:
                    continue
                
                # For each blocker, calculate net gain
                # Net Gain = Gain(y) - Loss(blocker) + (Interaction?)
                # Interaction: Removing blocker might break 3-APs involving y? No, y is not in mask yet.
                # Removing blocker breaks 3-APs containing blocker.
                # Adding y creates 3-APs containing y.
                # Are there 3-APs containing both y and blocker?
                # If so, Loss(blocker) counts them, and Gain(y) counts them.
                # Net change = Gain(y) - Loss(blocker).
                # Wait, if a 3-AP contains both y and blocker, it is destroyed by removing blocker, and created by adding y.
                # So it cancels out in the simple sum?
                # Let's verify:
                # Score = S(mask)
                # New Score = S(mask - {x} + {y})
                # S(mask - {x}) = S(mask) - Loss(x)
                # S(mask - {x} + {y}) = S(mask - {x}) + Gain_y_on_(mask-{x})
                # Gain_y_on_(mask-{x}) is the number of 3-APs formed by y using elements in mask-{x}.
                # This is slightly different from Gain(y) on mask.
                # If a 3-AP is (y, a, b) where a is the blocker x, then it's not counted in Gain_y_on_(mask-{x}).
                # So Gain_y_on_(mask-{x}) = Gain(y) - (number of 3-APs containing y and x).
                
                # To be safe and fast, we can just compute the actual score change for promising swaps.
                # But that requires count_3aps which is O(N^2).
                # Let's use the approximation: NetGain = Gain(y) - Loss(x).
                # If NetGain > 0, we do the swap and verify.
                
                for x in set(blockers):
                    checks += 1
                    if checks > 1000: break
                    loss_x = get_loss(mask, x)
                    gain_y = get_gain(mask, y) # Approximation
                    net = gain_y - loss_x
                    if net > best_swap_gain:
                        best_swap_gain = net
                        best_x_remove = x
                        best_y_add = y
            
            if best_x_remove != -1 and best_swap_gain > 0:
                # Execute swap
                mask &= ~(1 << best_x_remove)
                mask |= (1 << best_y_add)
                elements_in.remove(best_x_remove)
                elements_in.append(best_y_add)
                elements_out.remove(best_y_add)
                elements_out.append(best_x_remove)
                
                # Verify and update score
                new_score = count_3aps(mask)
                if new_score > current_score:
                    current_score = new_score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = mask
                else:
                    # Revert if it didn't help (shouldn't happen if approx was good, but safe)
                    mask &= ~(1 << best_y_add)
                    mask |= (1 << best_x_remove)
                    elements_in.remove(best_y_add)
                    elements_in.append(best_x_remove)
                    elements_out.remove(best_x_remove)
                    elements_out.append(best_y_add)
                    # Try kick
                    if random.random() < 0.5:
                         # Random removal to escape
                         if elements_in:
                             rem = random.choice(elements_in)
                             mask &= ~(1 << rem)
                             elements_in.remove(rem)
                             elements_out.append(rem)
                             current_score = count_3aps(mask)
                             improved = True
                continue

            if not improved:
                # Local optimum reached. Try a random kick
                if elements_in and random.random() < 0.8:
                    rem = random.choice(elements_in)
                    mask &= ~(1 << rem)
                    elements_in.remove(rem)
                    elements_out.append(rem)
                    current_score = count_3aps(mask)
                    # Do not update best here, just continue search
                else:
                    break

        return best_mask

    # Main Loop
    best_global_mask = 0
    best_global_score = 0
    
    # Strategy 1: Greedy Forward
    mask = 0
    for i in range(N):
        if is_valid_add(mask, i):
            mask |= (1 << i)
    score = count_3aps(mask)
    if score > best_global_score:
        best_global_score = score
        best_global_mask = mask
    
    if time.time() < deadline:
        improved_mask = local_search(mask, deadline - 5.0)
        s = count_3aps(improved_mask)
        if s > best_global_score:
            best_global_score = s
            best_global_mask = improved_mask

    # Strategy 2: Randomized Greedy
    for restart in range(10):
        if time.time() > deadline - 10.0:
            break
        random.seed(restart)
        order = list(range(N))
        random.shuffle(order)
        mask = 0
        for i in order:
            if is_valid_add(mask, i):
                mask |= (1 << i)
        score = count_3aps(mask)
        if score > best_global_score:
            best_global_score = score
            best_global_mask = mask
            
        if time.time() < deadline - 5.0:
            improved_mask = local_search(mask, deadline - 5.0)
            s = count_3aps(improved_mask)
            if s > best_global_score:
                best_global_score = s
                best_global_mask = improved_mask

    # Convert to 1-indexed list
    result = [i + 1 for i in range(N) if (best_global_mask >> i) & 1]
    if not result:
        result = [1]
    return sorted(result)

if __name__ == "__main__":
    # Test
    res = construct()
    print(f"Result size: {len(res)}")
    # Validate
    sel = set(res)
    has_4ap = False
    for a in range(1, 257):
        for d in range(1, (256 - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e > 256: break
            if a in sel and b in sel and c in sel and e in sel:
                has_4ap = True
                break
        if has_4ap: break
    print(f"Has 4-AP: {has_4ap}")
    # Count 3-AP
    count = 0
    for a in range(1, 257):
        if a not in sel: continue
        for d in range(1, (256 - a) // 2 + 1):
            b = a + d
            c = a + 2 * d
            if c > 256: break
            if b in sel and c in sel:
                count += 1
    print(f"3-AP Count: {count}")
