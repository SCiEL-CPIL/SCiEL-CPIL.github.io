import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Bitset representation
    # Value v in 1..512 maps to bit v-1 in an integer B
    
    def score_bitset(B):
        """Count distinct d > 0 such that 3-AP with diff d exists in bitset B."""
        count = 0
        # Max d is such that 1 + 2d <= 512 => 2d <= 511 => d <= 255
        for d in range(1, 256):
            # Check if there is a 3-AP with difference d
            # B & (B >> d) & (B >> 2*d)
            if (B & (B >> d) & (B >> (2 * d))) != 0:
                count += 1
        return count

    def list_to_bitset(S):
        B = 0
        for x in S:
            B |= (1 << (x - 1))
        return B

    def bitset_to_list(B):
        S = []
        # B is an integer, bits 0..511
        # Iterate through bits
        # Efficient way: while B:
        #   lsb = B & -B
        #   idx = lsb.bit_length() - 1
        #   S.append(idx + 1)
        #   B ^= lsb
        while B:
            lsb = B & -B
            idx = lsb.bit_length() - 1
            S.append(idx + 1)
            B ^= lsb
        return S

    rng = random.Random(42)
    
    best_set = None
    best_score = -1
    
    # Initial candidates
    candidates = []
    
    # 1. Simple AP
    candidates.append(list(range(1, 33)))
    
    # 2. Random sets
    for _ in range(20):
        s = rng.sample(range(1, 513), 32)
        candidates.append(s)
        
    # 3. Structured: Two blocks
    for start1 in range(1, 100, 10):
        for start2 in range(200, 400, 50):
            s1 = list(range(start1, start1 + 16))
            s2 = list(range(start2, start2 + 16))
            if max(s2) <= 512:
                candidates.append(s1 + s2)

    # Evaluate candidates
    for cand in candidates:
        if len(cand) != 32 or len(set(cand)) != 32:
            continue
        if min(cand) < 1 or max(cand) > 512:
            continue
        B = list_to_bitset(cand)
        sc = score_bitset(B)
        if sc > best_score:
            best_score = sc
            best_set = cand[:]

    # Simulated Annealing
    current_set = best_set[:]
    current_B = list_to_bitset(current_set)
    current_score = best_score
    
    max_iters = 200000
    current_temp = 10.0
    cooling = 0.9999
    
    iteration = 0
    while iteration < max_iters:
        if time.time() > deadline:
            break
        
        # Move: Swap one element
        idx = rng.randint(0, 31)
        old_val = current_set[idx]
        
        # Pick new value
        # To avoid duplicates, we can pick from 1..512 and check
        # Or pick from a pre-computed list of available values? 
        # 512 values is small. 
        new_val = rng.randint(1, 512)
        
        if new_val == old_val or new_val in current_set:
            iteration += 1
            continue
        
        # Update bitset
        # Remove old, add new
        new_B = current_B ^ (1 << (old_val - 1)) | (1 << (new_val - 1))
        
        new_score = score_bitset(new_B)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta > 0 or rng.random() < np.exp(delta / max(current_temp, 1e-10)):
            # Accept
            current_set[idx] = new_val
            current_B = new_B
            current_score = new_score
            
            if new_score > best_score:
                best_score = new_score
                best_set = current_set[:]
        else:
            # Reject
            pass # current_set and current_B remain unchanged
        
        iteration += 1
        if iteration % 100 == 0:
            current_temp *= cooling
            
    # Final Polish: Local Search (Hill Climbing)
    # Try to improve by swapping any pair
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        current_set = best_set[:]
        current_B = list_to_bitset(current_set)
        current_score = best_score
        
        for i in range(32):
            if time.time() > deadline - 5:
                break
            old_val = current_set[i]
            # Try swapping with other elements in the set
            for j in range(32):
                if i == j: continue
                other_val = current_set[j]
                # Swap i and j? No, that just rearranges. 
                # We need to change the set composition.
                # Try replacing old_val with a random new_val?
                # Or try replacing old_val with a value from 1..512 that is not in the set.
                
                # Let's do a more targeted local search:
                # Try removing old_val and adding a new_val that creates a new AP.
                # This is complex. Stick to random swaps in SA, but do a final pass of random swaps with no temperature (greedy).
                pass
        
        # Actually, let's just run a few more iterations of greedy hill climbing
        # where we only accept if score increases.
        
    # Run a final greedy pass
    greedy_set = best_set[:]
    greedy_B = list_to_bitset(greedy_set)
    greedy_score = best_score
    
    for _ in range(5000):
        if time.time() > deadline - 2:
            break
        idx = rng.randint(0, 31)
        old_val = greedy_set[idx]
        new_val = rng.randint(1, 512)
        if new_val == old_val or new_val in greedy_set:
            continue
        
        new_B = greedy_B ^ (1 << (old_val - 1)) | (1 << (new_val - 1))
        new_score = score_bitset(new_B)
        
        if new_score > greedy_score:
            greedy_set[idx] = new_val
            greedy_B = new_B
            greedy_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_set = greedy_set[:]

    return sorted(best_set)

if __name__ == "__main__":
    result = construct()
    assert len(result) == 32
    assert len(set(result)) == 32
    assert all(1 <= x <= 512 for x in result)
    # Verify score
    S_set = set(result)
    diffs = set()
    arr = sorted(result)
    for i in range(32):
        a = arr[i]
        for j in range(i+1, 32):
            c = arr[j]
            if (a + c) % 2 == 0:
                mid = (a + c) // 2
                if mid in S_set:
                    d = (c - a) // 2
                    if d > 0:
                        diffs.add(d)
    print(f"Score: {len(diffs)}")
    print(f"Set: {arr}")
