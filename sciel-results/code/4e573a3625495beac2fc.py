import time
import random
import math

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # safety margin

    # For k=4, g=10: search for bipartite Cayley graph on Z_n x {0,1}
    # Total vertices = 2*n, want 2*n = 320 => n = 160
    # 4 generators, no non-backtracking walk of length 2,4,6,8 closes to 0
    
    if k != 4 or g != 10:
        # Fallback for other parameters: LUW construction
        return _luw_construct(k, g, deadline)
    
    # Precompute all non-backtracking sequences of even length 2,4,6,8
    # Steps: 0,1,2,3 are +s_i; 4,5,6,7 are -s_i
    # Inverse of step i is step i^4 (i.e., 0<->4, 1<->5, 2<->6, 3<->7)
    
    def generate_nb_sequences(max_len):
        """Generate all non-backtracking sequences up to max_len, return dict {length: [sequences]}"""
        # Start with length-1 sequences
        current = [(i,) for i in range(8)]
        result = {}
        for L in range(2, max_len + 1):
            next_current = []
            for seq in current:
                last = seq[-1]
                inv = last ^ 4
                for step in range(8):
                    if step != inv:
                        next_current.append(seq + (step,))
            current = next_current
            if L % 2 == 0:
                result[L] = current
        return result
    
    seqs_by_len = generate_nb_sequences(8)
    
    # For each sequence, precompute the coefficient vector
    # A sequence (s1, s2, ..., sL) has net displacement = sum of +/- s_i
    # Represent as: for each generator j (0..3), net count of +s_j minus count of -s_j
    # So coeff[j] = (count of step j) - (count of step j+4)
    
    def compute_coefficients(seq):
        coeffs = [0, 0, 0, 0]
        for step in seq:
            if step < 4:
                coeffs[step] += 1
            else:
                coeffs[step - 4] -= 1
        return tuple(coeffs)
    
    # Precompute all coefficient vectors for each length
    coeffs_by_len = {}
    for L, seqs in seqs_by_len.items():
        coeffs_by_len[L] = set(compute_coefficients(s) for s in seqs)
    
    # For a given n and generators (s0, s1, s2, s3), check if any coefficient vector
    # gives sum(c_i * s_i) == 0 mod n
    
    def check_valid(n, s):
        """Check if generators s = (s0,s1,s2,s3) on Z_n give girth >= 10"""
        for L in [2, 4, 6, 8]:
            for c in coeffs_by_len[L]:
                total = c[0]*s[0] + c[1]*s[1] + c[2]*s[2] + c[3]*s[3]
                if total % n == 0:
                    return False
        return True
    
    # Search for n starting from 160
    random.seed(42)
    
    for n in range(160, 181):
        if time.time() > deadline - 20:
            break
        
        # Fix s[0] = 1 for symmetry breaking
        # Try to find s[1], s[2], s[3]
        
        # First try some structured candidates
        # Then random search
        found = False
        
        # Number of trials: be aggressive for n=160
        if n == 160:
            max_trials = 200000
        elif n <= 165:
            max_trials = 50000
        else:
            max_trials = 10000
        
        trial_count = 0
        while trial_count < max_trials:
            if time.time() > deadline - 10:
                break
            
            # Generate 3 random distinct values in 1..n-1
            # Exclude 1 and n-1 (inverse of 1)
            # Exclude pairs that are inverses of each other
            S = [1]
            attempts = 0
            while len(S) < 4 and attempts < 100:
                x = random.randint(2, n - 2)
                conflict = False
                for y in S:
                    if x == y or x == n - y:
                        conflict = True
                        break
                if not conflict:
                    S.append(x)
                attempts += 1
            
            if len(S) < 4:
                trial_count += 1
                continue
            
            if check_valid(n, tuple(S)):
                # Construct the bipartite Cayley graph
                edges = []
                for u in range(n):
                    for s in S:
                        v = (u + s) % n
                        edges.append((u, v + n))
                return 2 * n, edges
            
            trial_count += 1
    
    # Fallback to LUW construction
    return _luw_construct(k, g, deadline)


def _luw_construct(k, g, deadline):
    """LUW construction fallback"""
    start_time = time.time()
    
    q = k
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def gf_add(x, y): return x ^ y
        def gf_sub(x, y): return x ^ y
        def gf_mul(x, y): return table[x][y]
    else:
        def gf_add(x, y): return (x + y) % q
        def gf_sub(x, y): return (x - y) % q
        def gf_mul(x, y): return x * y % q

    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = gf_add(x[c], gf_mul(y[a], x[b]))
                else:
                    y[c] = gf_sub(x[c], gf_mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    queue = [0]
    head = 0
    
    while head < len(order):
        if time.time() > deadline:
            break
        v_idx = queue[head]
        head += 1
        v = order[v_idx]
        for other in neighbours(v):
            if other not in index:
                index[other] = len(order)
                order.append(other)
                queue.append(len(order) - 1)
            if v[0] == 0:
                edges.append((v_idx, index[other]))
                
    return len(order), edges
