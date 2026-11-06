import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    # Known record is n=50
    n = 50
    # Vertices 0..49
    # Differences mod 51
    # Q: Quadratic Residues mod 51
    # N: Quadratic Non-Residues mod 51
    
    # Calculate Q and N
    Q = set()
    for x in range(1, 51):
        Q.add((x * x) % 51)
    Q.discard(0)
    # Q should have 16 elements
    N = set(range(1, 51)) - Q
    # N should have 16 elements
    
    # We need to partition Q into Q1, Q2 and N into N1, N2
    # such that Q1, Q2, N1, N2 are triangle-free (in the additive group sense)
    # i.e., no x, y, z in S such that x+y+z == 0 mod 51
    
    def is_triangle_free(S):
        S_list = list(S)
        n_s = len(S_list)
        for i in range(n_s):
            for j in range(i, n_s):
                for k in range(j, n_s):
                    if (S_list[i] + S_list[j] + S_list[k]) % 51 == 0:
                        return False
        return True

    # Try to find partitions
    # Randomized search
    best_colors = None
    for _ in range(10000):
        if time.time() > deadline:
            break
        # Randomly split Q into Q1, Q2
        Q_list = list(Q)
        random.shuffle(Q_list)
        split_q = random.randint(0, len(Q_list))
        Q1 = set(Q_list[:split_q])
        Q2 = set(Q_list[split_q:])
        
        if not is_triangle_free(Q1) or not is_triangle_free(Q2):
            continue
            
        # Randomly split N into N1, N2
        N_list = list(N)
        random.shuffle(N_list)
        split_n = random.randint(0, len(N_list))
        N1 = set(N_list[:split_n])
        N2 = set(N_list[split_n:])
        
        if not is_triangle_free(N1) or not is_triangle_free(N2):
            continue
            
        # Valid partition found
        # Map colors
        # 0: Q1, 1: Q2, 2: N1, 3: N2
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                d = (i - j) % 51
                if d in Q1:
                    colors.append(0)
                elif d in Q2:
                    colors.append(1)
                elif d in N1:
                    colors.append(2)
                else:
                    colors.append(3)
        best_colors = colors
        break
        
    if best_colors is None:
        # Fallback to n=32 if search fails
        # Use the seed program logic
        def mul(a, b):
            r = 0
            for i in range(4):
                if b >> i & 1:
                    r ^= a << i
            for i in range(7, 3, -1):
                if r >> i & 1:
                    r ^= 0b10011 << (i - 4)
            return r
        log, x = {}, 1
        for e in range(15):
            log[x] = e
            x = mul(x, 2)
        colors = []
        for i in range(32):
            for j in range(i + 1, 32):
                colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
        return {'n': 32, 'colors': colors}
        
    return {'n': n, 'colors': best_colors}
