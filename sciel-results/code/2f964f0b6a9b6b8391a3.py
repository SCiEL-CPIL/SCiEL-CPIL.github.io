import time
import itertools
import random

# EVOLVE-BLOCK-START

def _get_classic_order(n):
    """
    Get the order of (n-1)-permutations as they appear in the classic superpermutation of n-1 symbols.
    """
    if n == 1:
        return [()]
    if n == 2:
        return [(1,), (2,)] # Actually S_1 is "1", perm is (1,). Wait.
    
    # Build classic superperm for n-1
    # S_1 = "1"
    # S_2 = "121"
    # S_3 = "123121321"
    
    word = [1]
    for m in range(2, n):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = 0
            for o in range(min(len(new), len(block)), -1, -1):
                if o > 0 and new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
                elif o == 0:
                    overlap = 0
                    break
            new += block[overlap:]
        word = new
        
    # Extract perms of length n-1 from word
    perms, seen = [], set()
    m = n - 1
    for i in range(len(word) - m + 1):
        window = tuple(word[i:i + m])
        if len(set(window)) == m and window not in seen:
            seen.add(window)
            perms.append(window)
    return perms

def _overlap(u, v):
    """
    Calculate overlap between block B(u) and B(v).
    B(u) ends with u. B(v) starts with v.
    Overlap is longest suffix of u that is prefix of v.
    """
    max_ov = min(len(u), len(v))
    for o in range(max_ov, 0, -1):
        if u[-o:] == v[:o]:
            return o
    return 0

def _build_word(order, n):
    """
    Build the superpermutation word from the given order of (n-1)-perms.
    """
    if not order:
        return ""
    
    m = n
    blocks = [list(p) + [m] + list(p) for p in order]
    
    word = list(blocks[0])
    for i in range(1, len(blocks)):
        prev_block = blocks[i-1]
        curr_block = blocks[i]
        # Calculate overlap
        max_ov = min(len(prev_block), len(curr_block))
        ov = 0
        for o in range(max_ov, 0, -1):
            if prev_block[-o:] == curr_block[:o]:
                ov = o
                break
        word += curr_block[ov:]
        
    return "".join(map(str, word))

def construct(n):
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "123412134212343142341234"
    
    # For n >= 5, use optimization
    if n > 5:
        # Get the classic order of (n-1) perms
        order = _get_classic_order(n)
        N = len(order)
        
        # Precompute overlap matrix for efficiency in local search
        # ov[i][j] = overlap(order[i], order[j])
        # Note: order is a list of tuples.
        ov_matrix = [[0]*N for _ in range(N)]
        for i in range(N):
            for j in range(N):
                if i != j:
                    ov_matrix[i][j] = _overlap(order[i], order[j])
        
        # Current cost: sum of (len(block) - overlap)
        # len(block) is constant 2*(n-1)+1
        # We want to maximize sum of overlaps.
        
        def get_cost(seq):
            cost = 0
            for i in range(len(seq)-1):
                cost += ov_matrix[seq[i]][seq[i+1]]
            return cost
        
        best_seq = list(range(N))
        best_cost = get_cost(best_seq)
        
        # Local Search
        # 2-Opt
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for i in range(N-1):
                if time.time() > deadline - 5.0:
                    break
                for j in range(i+1, N):
                    # Reverse seq[i+1..j]
                    # Old edges: (i, i+1), (j, j+1)
                    # New edges: (i, j), (i+1, j+1)
                    old_cost = 0
                    new_cost = 0
                    
                    if i > 0: pass # Start node fixed? No, path is open.
                    # Edges affected:
                    # 1. i -> i+1
                    # 2. j -> j+1 (if j+1 < N)
                    # New:
                    # 1. i -> j
                    # 2. i+1 -> j+1 (if j+1 < N)
                    
                    # Note: The cost function sums overlaps.
                    # We want to maximize.
                    
                    # Edge (i, i+1) is removed.
                    old_cost += ov_matrix[best_seq[i]][best_seq[i+1]]
                    # Edge (j, j+1) is removed if j+1 < N
                    if j + 1 < N:
                        old_cost += ov_matrix[best_seq[j]][best_seq[j+1]]
                        
                    # Edge (i, j) is added.
                    new_cost += ov_matrix[best_seq[i]][best_seq[j]]
                    # Edge (i+1, j+1) is added if j+1 < N
                    if j + 1 < N:
                        new_cost += ov_matrix[best_seq[i+1]][best_seq[j+1]]
                        
                    if new_cost > old_cost:
                        # Perform reversal
                        best_seq[i+1:j+1] = best_seq[i+1:j+1][::-1]
                        best_cost += new_cost - old_cost
                        improved = True
                        
        # Or-Opt (move segment of length 1, 2, 3)
        if time.time() < deadline - 5.0:
            improved = True
            while improved and time.time() < deadline - 2.0:
                improved = False
                for seg_len in [1, 2, 3]:
                    if time.time() > deadline - 2.0:
                        break
                    for i in range(N - seg_len):
                        if time.time() > deadline - 2.0:
                            break
                        seg = best_seq[i:i+seg_len]
                        # Remove seg
                        rem = best_seq[:i] + best_seq[i+seg_len:]
                        if not rem:
                            continue
                        for j in range(len(rem) - seg_len + 1):
                            # Insert seg at j
                            # This is O(N) to check all j, and O(N^2) total per pass.
                            # Might be slow for N=720.
                            # N=720, N^2 = 518k. 3 seg_lens. 1.5M ops per pass.
                            # Python might be too slow for many passes.
                            # Let's limit passes or use a smarter check.
                            # Actually, 150s is plenty for a few passes of Or-Opt if optimized.
                            # But let's stick to 2-opt primarily as it's O(N^2) per pass.
                            # 2-opt is O(N^2) = 500k checks per pass.
                            # Each check is O(1).
                            # 500k checks in Python takes ~0.1-0.5s.
                            # So we can do ~100-1000 passes of 2-opt.
                            # Or-Opt is O(N^2 * seg_len) ~ 1.5M checks.
                            # So maybe 10-50 passes.
                            
                            # Let's just do 2-opt for now, it's safer.
                            pass
                            
        # Construct final word
        return _build_word([order[i] for i in best_seq], n)
        
    else:
        # Fallback for n=5
        # Classic is 153. Optimal is 153? 
        # S_5 classic: 1+2+6+24+120 = 153.
        # Is there a shorter one?
        # Lower bound for n=5 is 153?
        # Actually, for n=5, the classic is optimal.
        return _build_word(_get_classic_order(n), n)

# EVOLVE-BLOCK-END
