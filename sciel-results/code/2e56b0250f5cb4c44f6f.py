"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from itertools import combinations

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline, 5s margin
    n = 128
    N = 128  # elements 1..128, indexed 0..127
    
    # Precompute LCM table
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            lcm_table[i][j] = _lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]
    
    # Precompute bad triples conflicts
    # bad_partners[x-1][a-1] is a bitmask of b-1 such that {x, a, b} is a bad triple.
    bad_partners = [[0] * N for _ in range(N)]
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            L_ab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                if lcm_table[a][c] == L_ab and lcm_table[b][c] == L_ab:
                    bad_partners[a-1][b-1] |= (1 << (c-1))
                    bad_partners[b-1][a-1] |= (1 << (c-1))
                    bad_partners[a-1][c-1] |= (1 << (b-1))
                    bad_partners[c-1][a-1] |= (1 << (b-1))
                    bad_partners[b-1][c-1] |= (1 << (a-1))
                    bad_partners[c-1][b-1] |= (1 << (a-1))
    
    def can_add(x, mask):
        """Check if element x (1-indexed) can be added to set represented by bitmask mask."""
        xi = x - 1
        m = mask
        while m:
            lb = m & (-m)
            ai = lb.bit_length() - 1
            if (bad_partners[xi][ai] & mask) != 0:
                return False
            m ^= lb
        return True
    
    def conflict_count(x, mask):
        """Count how many elements in mask would form a bad triple with x."""
        xi = x - 1
        count = 0
        m = mask
        while m:
            lb = m & (-m)
            ai = lb.bit_length() - 1
            count += (bad_partners[xi][ai] & mask).bit_count()
            m ^= lb
        return count
    
    def is_valid_mask(mask):
        """Verify the mask is valid. Used for final check."""
        m = mask
        while m:
            la = m & (-m)
            ai = la.bit_length() - 1
            m ^= la
            m2 = m
            while m2:
                lb = m2 & (-m2)
                bi = lb.bit_length() - 1
                m2 ^= lb
                bp = bad_partners[ai][bi]
                bp &= ~(la | lb)
                if (bp & mask) != 0:
                    return False
        return True
    
    # Greedy construction with multiple orderings
    def greedy(order):
        mask = 0
        for x in order:
            if can_add(x, mask):
                mask |= (1 << (x - 1))
        return mask
    
    best_mask = 0
    best_size = 0
    
    # Define various orderings based on structural properties
    nums = list(range(1, n + 1))
    
    # Compute omega (total prime factors) for sorting
    def omega(x):
        c = 0
        n = x
        d = 2
        while d * d <= n:
            while n % d == 0:
                c += 1
                n //= d
            d += 1
        if n > 1:
            c += 1
        return c
        
    omega_cache = {x: omega(x) for x in nums}
    
    orders = [
        nums,
        nums[::-1],
        sorted(nums, key=lambda x: x.bit_count()),
        sorted(nums, key=lambda x: -x.bit_count()),
        sorted(nums, key=lambda x: omega_cache[x]),
        sorted(nums, key=lambda x: -omega_cache[x]),
        sorted(nums, key=lambda x: (omega_cache[x], x.bit_count())),
        sorted(nums, key=lambda x: (x.bit_count(), omega_cache[x])),
    ]
    
    for _ in range(8):
        rand_order = nums[:]
        random.shuffle(rand_order)
        orders.append(rand_order)
    
    for order in orders:
        if time.time() > deadline:
            break
        mask = greedy(order)
        sz = mask.bit_count()
        if sz > best_size:
            best_size = sz
            best_mask = mask
    
    # Conflict-driven repair local search
    def repair_search(start_mask, max_iters=200000):
        """Local search with conflict-driven swaps."""
        current_mask = start_mask
        current_size = current_mask.bit_count()
        best_local_mask = current_mask
        best_local_size = current_size
        
        no_improve = 0
        max_no_improve = 5000
        
        for it in range(max_iters):
            if time.time() >= deadline:
                break
            if no_improve > max_no_improve:
                break
            
            # Try to add an element
            not_in = ((1 << N) - 1) & ~current_mask
            if not_in:
                # Find element with minimum conflict count
                best_x = -1
                best_conf = float('inf')
                m = not_in
                # Sample up to 20 candidates for speed
                sample_count = min(20, m.bit_count())
                for _ in range(sample_count):
                    idx = random.randint(0, m.bit_count() - 1)
                    tmp = m
                    for _2 in range(idx):
                        tmp &= tmp - 1
                    x_idx = (tmp & (-tmp)).bit_length() - 1
                    x = x_idx + 1
                    conf = conflict_count(x, current_mask)
                    if conf < best_conf:
                        best_conf = conf
                        best_x = x
                
                if best_x != -1 and best_conf == 0:
                    new_mask = current_mask | (1 << (best_x - 1))
                    new_size = new_mask.bit_count()
                    if new_size > best_local_size:
                        best_local_size = new_size
                        best_local_mask = new_mask
                    current_mask = new_mask
                    current_size = new_size
                    no_improve = 0
                    continue
            
            # Try swap: remove one element, add another
            if current_mask:
                # Find element with highest "removal benefit" (most conflicts it causes)
                # Approximate: element whose removal allows most additions
                m_in = current_mask
                sample_in = min(10, m_in.bit_count())
                best_remove = -1
                best_remove_score = -1
                
                for _ in range(sample_in):
                    idx = random.randint(0, m_in.bit_count() - 1)
                    tmp = m_in
                    for _2 in range(idx):
                        tmp &= tmp - 1
                    x_idx = (tmp & (-tmp)).bit_length() - 1
                    x = x_idx + 1
                    
                    # Score: how many elements in not_in could be added if x were removed
                    mask_without = current_mask & ~(1 << x_idx)
                    not_in_new = ((1 << N) - 1) & ~mask_without
                    addable = 0
                    m = not_in_new
                    sample_out = min(15, m.bit_count())
                    for _3 in range(sample_out):
                        idx2 = random.randint(0, m.bit_count() - 1)
                        tmp2 = m
                        for _4 in range(idx2):
                            tmp2 &= tmp2 - 1
                        y_idx = (tmp2 & (-tmp2)).bit_length() - 1
                        y = y_idx + 1
                        if can_add(y, mask_without):
                            addable += 1
                    
                    score = addable
                    if score > best_remove_score:
                        best_remove_score = score
                        best_remove = x_idx
                
                if best_remove != -1 and best_remove_score > 0:
                    mask_without = current_mask & ~(1 << best_remove)
                    not_in_new = ((1 << N) - 1) & ~mask_without
                    
                    # Find best element to add
                    best_add = -1
                    m = not_in_new
                    sample_out = min(20, m.bit_count())
                    for _ in range(sample_out):
                        idx = random.randint(0, m.bit_count() - 1)
                        tmp = m
                        for _2 in range(idx):
                            tmp &= tmp - 1
                        y_idx = (tmp & (-tmp)).bit_length() - 1
                        y = y_idx + 1
                        if can_add(y, mask_without):
                            best_add = y_idx
                            break
                    
                    if best_add != -1:
                        new_mask = mask_without | (1 << best_add)
                        new_size = new_mask.bit_count()
                        if new_size > best_local_size:
                            best_local_size = new_size
                            best_local_mask = new_mask
                        current_mask = new_mask
                        current_size = new_size
                        no_improve = 0
                        continue
            
            no_improve += 1
        
        return best_local_mask, best_local_size
    
    # Run repair search from best greedy solution
    if time.time() < deadline - 5:
        mask, sz = repair_search(best_mask)
        if sz > best_size:
            best_size = sz
            best_mask = mask
    
    # Try repair search from a few random initial solutions
    random_starts = 4
    for _ in range(random_starts):
        if time.time() >= deadline - 10:
            break
        # Random initial set with bias toward low-conflict elements
        init_mask = 0
        # Sort by conflict potential (elements with fewer bad partners first)
        conflict_potential = {}
        for x in range(1, n + 1):
            total = 0
            for a in range(1, n + 1):
                if a != x:
                    total += bad_partners[x-1][a-1].bit_count()
            conflict_potential[x] = total
        
        sorted_by_conf = sorted(range(1, n + 1), key=lambda x: conflict_potential[x])
        random.shuffle(sorted_by_conf)
        
        for x in sorted_by_conf:
            if random.random() < 0.6:
                if can_add(x, init_mask):
                    init_mask |= (1 << (x - 1))
        
        mask, sz = repair_search(init_mask, max_iters=100000)
        if sz > best_size:
            best_size = sz
            best_mask = mask
    
    # Final local search: try to add any element greedily
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for x in range(1, n + 1):
                if not (best_mask & (1 << (x - 1))):
                    if can_add(x, best_mask):
                        best_mask |= (1 << (x - 1))
                        best_size += 1
                        improved = True
    
    # Convert mask to list
    result = [x for x in range(1, n + 1) if best_mask & (1 << (x - 1))]
    
    # Final verification
    assert len(result) > 0
    for a, b, c in combinations(result, 3):
        assert not (_lcm(a, b) == _lcm(a, c) == _lcm(b, c)), f"Bad triple: {a},{b},{c}"
    
    return result
# EVOLVE-BLOCK-END
