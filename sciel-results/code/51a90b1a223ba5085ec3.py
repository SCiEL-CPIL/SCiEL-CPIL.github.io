import time
import random

N = 256
TOTAL_TIME = 85.0

def _precompute():
    """
    Precompute:
    - For each x, the list of (y, z) pairs such that {x, y, z} forms a 3-AP.
    - For each x, the list of (y, z, w) triples such that {x, y, z, w} forms a 4-AP.
    """
    three_ap_pairs = [[] for _ in range(N + 1)]
    four_ap_triples = [[] for _ in range(N + 1)]

    for x in range(1, N + 1):
        # x as first: x, x+d, x+2d
        for d in range(1, (N - x) // 2 + 1):
            y = x + d
            z = x + 2 * d
            if z <= N:
                three_ap_pairs[x].append((y, z))
        # x as second: x-d, x, x+d
        for d in range(1, min(x - 1, N - x) + 1):
            y = x - d
            z = x + d
            if y >= 1 and z <= N:
                three_ap_pairs[x].append((y, z))
        # x as third: x-2d, x-d, x
        for d in range(1, x // 2 + 1):
            y = x - 2 * d
            z = x - d
            if y >= 1 and z <= N:
                three_ap_pairs[x].append((y, z))

        # x as 1st in 4-AP: x, x+d, x+2d, x+3d
        for d in range(1, (N - x) // 3 + 1):
            y = x + d
            z = x + 2 * d
            w = x + 3 * d
            if w <= N:
                four_ap_triples[x].append((y, z, w))
        # x as 2nd: x-d, x, x+d, x+2d
        for d in range(1, min(x - 1, (N - x) // 2) + 1):
            y = x - d
            z = x + d
            w = x + 2 * d
            if y >= 1 and w <= N:
                four_ap_triples[x].append((y, z, w))
        # x as 3rd: x-2d, x-d, x, x+d
        for d in range(1, min(x // 2, N - x) + 1):
            y = x - 2 * d
            z = x - d
            w = x + d
            if y >= 1 and w <= N:
                four_ap_triples[x].append((y, z, w))
        # x as 4th: x-3d, x-2d, x-d, x
        for d in range(1, (x - 1) // 3 + 1):
            y = x - 3 * d
            z = x - 2 * d
            w = x - d
            if y >= 1:
                four_ap_triples[x].append((y, z, w))

    return three_ap_pairs, four_ap_triples

THREE_AP_PAIRS, FOUR_AP_TRIPLES = _precompute()

def _count_3aps(mask):
    """
    Count the number of 3-APs in the set represented by the bitmask.
    We count each distinct 3-AP once.
    A 3-AP is (a, b, c) with a < b < c and b-a = c-b.
    Using the precomputed pairs for x, we count if x is in the set and its pair (y, z) is in the set.
    However, this counts each AP 3 times (once for each element).
    To be precise, let's just iterate over the canonical form (a, b, c) with a < b < c.
    But for speed, we can use the precomputed structure.
    Actually, the precomputed `three_ap_pairs` contains all permutations.
    Let's just iterate a, b, c directly for correctness in counting, or fix the counting.
    Wait, the task says "Maximize the number of distinct three-term arithmetic progressions (positive difference, counted once)".
    So (a, b, c) with a < b < c is one progression.
    My `three_ap_pairs` has (y, z) for each x.
    If I sum over x in A: count of (y, z) in A, I get 3 * (number of 3-APs).
    So I can just divide by 3.
    """
    count = 0
    for x in range(1, N + 1):
        if (mask >> x) & 1:
            for y, z in THREE_AP_PAIRS[x]:
                if (mask >> y) & 1 and (mask >> z) & 1:
                    count += 1
    return count // 3

def _creates_4ap(x, mask):
    """Check if adding x to the set (represented by mask) creates a 4-AP."""
    for y, z, w in FOUR_AP_TRIPLES[x]:
        if (mask >> y) & 1 and (mask >> z) & 1 and (mask >> w) & 1:
            return True
    return False

def _gain_3ap(x, mask):
    """Calculate the increase in 3-AP count if x is added to the set (represented by mask)."""
    gain = 0
    for y, z in THREE_AP_PAIRS[x]:
        if (mask >> y) & 1 and (mask >> z) & 1:
            gain += 1
    return gain // 3 # Since each AP is counted 3 times total, adding x completes 3 * (new APs containing x)
    # Wait. If x is added, it forms new APs with pairs (y, z) already in A.
    # Each such pair (y, z) forms exactly one AP with x.
    # So the gain is exactly the number of such pairs.
    # The previous count was 3 * total_APs.
    # New count will be 3 * (total_APs + new_APs).
    # So gain in the "raw count" is 3 * new_APs.
    # My function returns gain // 3, which is new_APs.
    # This is consistent.

def _greedy_construct(deadline):
    """
    Greedily construct a set A.
    """
    mask = 0
    A_set = set()
    
    # We want to add elements that maximize 3-AP gain.
    # To avoid getting stuck in a local optimum or a "bad" structure, we can use a slight randomization or look-ahead.
    # For now, simple greedy.
    
    while time.time() < deadline - 10: # Leave 10s for local search
        best_x = -1
        best_gain = -1
        
        # We can restrict candidates to speed up?
        # N is 256, so checking all is fine.
        for x in range(1, N + 1):
            if (mask >> x) & 1:
                continue
            
            if _creates_4ap(x, mask):
                continue
            
            gain = _gain_3ap(x, mask)
            
            if gain > best_gain:
                best_gain = gain
                best_x = x
        
        if best_x == -1:
            break
            
        mask |= (1 << best_x)
        A_set.add(best_x)
        
    return A_set, mask

def _local_search(A, deadline):
    """
    Iterated Local Search with swaps.
    """
    A_set = set(A)
    mask = 0
    for x in A_set:
        mask |= (1 << x)
        
    current_count = _count_3aps(mask)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try all single swaps: remove x, add y
        # Optimization: Precompute loss of removing x?
        # Loss of x = number of 3-APs containing x.
        # But we need to check if adding y is valid.
        
        # Let's just do a randomized local search to save time and avoid O(N^2) if N grows, though N=256 is small.
        # Actually, O(N^2) = 65k. _count_3aps is ~2000 ops. 65k * 2000 = 130M ops. Might be slow in Python.
        # Better: Incremental update.
        
        # Let's try a few random swaps per pass.
        for _ in range(50): # 50 random swaps per pass
            if time.time() > deadline:
                break
                
            if len(A_set) == 0:
                break
                
            x = random.choice(list(A_set))
            y = random.randint(1, N)
            
            if y in A_set:
                continue
            
            # Check if we can remove x and add y
            # New mask = mask ^ (1<<x) | (1<<y)
            # But we must ensure y doesn't create 4-AP in (A - {x})
            
            mask_temp = mask & ~(1 << x)
            
            if _creates_4ap(y, mask_temp):
                continue
            
            new_mask = mask_temp | (1 << y)
            new_count = _count_3aps(new_mask)
            
            if new_count > current_count:
                A_set.remove(x)
                A_set.add(y)
                mask = new_mask
                current_count = new_count
                improved = True
                break # Restart the pass with the new set
                
    return A_set, mask

def _verify(A):
    """Verify no 4-APs."""
    A_set = set(A)
    for a in A:
        for d in range(1, N + 1):
            if (a+d) in A_set and (a+2*d) in A_set and (a+3*d) in A_set:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + TOTAL_TIME - 5.0 # 5s safety margin
    
    # 1. Greedy Construction
    A_greedy, mask_greedy = _greedy_construct(deadline)
    
    # 2. Local Search
    A_ls, mask_ls = _local_search(A_greedy, deadline)
    
    # 3. Verify
    if not _verify(A_ls):
        # Fallback
        return [1]
        
    return sorted(A_ls)

if __name__ == "__main__":
    # Quick test
    res = construct()
    print(f"Result size: {len(res)}")
    print(f"3-AP count: {_count_3aps(sum(1<<x for x in res))}")
