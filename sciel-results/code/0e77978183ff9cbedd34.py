import random
import time
import math
import itertools

def standard_construct(n):
    word = [1]
    for m in range(2, n + 1):
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
            # Find max overlap
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return "".join(map(str, word))

def construct(n):
    # Get the baseline
    baseline_word = standard_construct(n)
    baseline_len = len(baseline_word)
    
    if n <= 4:
        return baseline_word
        
    # For n=7, try to improve with SA
    if n == 7:
        start_time = time.time()
        deadline = start_time + 100  # Leave 20s margin
        
        try:
            # Generate all permutations
            perms = list(itertools.permutations(range(1, n + 1)))
            M = len(perms) # 5040
            
            # Precompute overlaps
            # O[i][j] = overlap length between perms[i] and perms[j]
            # Overlap is max k such that perms[i][-k:] == perms[j][:k]
            # We can store this in a list of lists or a flat array
            # For speed, let's use a list of arrays or just lists
            overlaps = [[0]*M for _ in range(M)]
            
            # Optimize overlap calculation
            # Since n=7, we can just loop k from 1 to n-1
            for i in range(M):
                pi = perms[i]
                for j in range(M):
                    pj = perms[j]
                    max_ov = 0
                    # Check k from 1 to 6
                    # pi[-1:] == pj[:1], etc.
                    # We can break early if found? No, we want max.
                    # So check from 6 down to 1.
                    for k in range(n-1, 0, -1):
                        if pi[-k:] == pj[:k]:
                            max_ov = k
                            break
                    overlaps[i][j] = max_ov
            
            # Cost matrix W[i][j] = n - overlaps[i][j]
            # We want to minimize sum of W[seq[k]][seq[k+1]]
            
            # Initialize sequence
            seq = list(range(M))
            random.shuffle(seq)
            
            # Calculate initial energy
            # E = sum_{k=0}^{M-2} (n - overlaps[seq[k]][seq[k+1]])
            # Note: The total length is n + sum of costs.
            # So minimizing sum of costs minimizes length.
            
            def calc_energy(seq):
                e = 0
                for k in range(M - 1):
                    e += n - overlaps[seq[k]][seq[k+1]]
                return e
            
            current_energy = calc_energy(seq)
            best_energy = current_energy
            best_seq = seq.copy()
            
            # Simulated Annealing
            T = 10.0
            T_min = 0.01
            cooling = 0.99999 # Adjust to fit time
            # Estimate steps: 100s / time_per_step
            # Time per step is very small, maybe 100ns-1us?
            # Let's just run until deadline.
            
            while time.time() < deadline:
                # Swap two random positions
                i = random.randint(0, M - 1)
                j = random.randint(0, M - 1)
                if i == j:
                    continue
                
                # Calculate delta E
                # If i and j are adjacent, handle separately
                if abs(i - j) == 1:
                    # Swap i and i+1
                    if i == 0:
                        # seq[0], seq[1] -> seq[1], seq[0]
                        # Remove: W[0,1]
                        # Add: W[1,0] + W[1,2] - W[0,2] ? No.
                        # Old: W[0,1] + W[1,2]
                        # New: W[1,0] + W[0,2]
                        # Delta = (W[1,0] + W[0,2]) - (W[0,1] + W[1,2])
                        # Note: W is cost.
                        w01 = n - overlaps[seq[0]][seq[1]]
                        w12 = n - overlaps[seq[1]][seq[2]]
                        w10 = n - overlaps[seq[1]][seq[0]]
                        w02 = n - overlaps[seq[0]][seq[2]]
                        delta = (w10 + w02) - (w01 + w12)
                    elif i == M - 2:
                        # seq[M-2], seq[M-1] -> seq[M-1], seq[M-2]
                        # Old: W[M-3, M-2] + W[M-2, M-1]
                        # New: W[M-3, M-1] + W[M-1, M-2]
                        w_m3_m2 = n - overlaps[seq[M-3]][seq[M-2]]
                        w_m2_m1 = n - overlaps[seq[M-2]][seq[M-1]]
                        w_m3_m1 = n - overlaps[seq[M-3]][seq[M-1]]
                        w_m1_m2 = n - overlaps[seq[M-1]][seq[M-2]]
                        delta = (w_m3_m1 + w_m1_m2) - (w_m3_m2 + w_m2_m1)
                    else:
                        # i, i+1
                        # Old: W[i-1, i] + W[i, i+1] + W[i+1, i+2]
                        # New: W[i-1, i+1] + W[i+1, i] + W[i, i+2]
                        w_im1_i = n - overlaps[seq[i-1]][seq[i]]
                        w_i_ip1 = n - overlaps[seq[i]][seq[i+1]]
                        w_ip1_ip2 = n - overlaps[seq[i+1]][seq[i+2]]
                        
                        w_im1_ip1 = n - overlaps[seq[i-1]][seq[i+1]]
                        w_ip1_i = n - overlaps[seq[i+1]][seq[i]]
                        w_i_ip2 = n - overlaps[seq[i]][seq[i+2]]
                        
                        delta = (w_im1_ip1 + w_ip1_i + w_i_ip2) - (w_im1_i + w_i_ip1 + w_ip1_ip2)
                else:
                    # Non-adjacent
                    # Old: W[i-1, i] + W[i, i+1] + W[j-1, j] + W[j, j+1]
                    # New: W[i-1, j] + W[j, i+1] + W[j-1, i] + W[i, j+1]
                    # Handle boundaries
                    # i-1, i+1, j-1, j+1
                    # If i=0, i-1 is M-1? No, sequence is linear, not circular.
                    # So if i=0, there is no i-1.
                    
                    # Let's use a helper to get cost or 0 if out of bounds
                    def get_cost(a, b):
                        if a < 0 or a >= M or b < 0 or b >= M:
                            return 0
                        return n - overlaps[a][b]
                    
                    # This is slow. Let's inline with checks.
                    
                    # Edges to remove
                    c1 = n - overlaps[seq[i-1]][seq[i]] if i > 0 else 0
                    c2 = n - overlaps[seq[i]][seq[i+1]] if i < M-1 else 0
                    c3 = n - overlaps[seq[j-1]][seq[j]] if j > 0 else 0
                    c4 = n - overlaps[seq[j]][seq[j+1]] if j < M-1 else 0
                    
                    # Edges to add
                    # seq[i-1] connects to seq[j]
                    # seq[j] connects to seq[i+1]
                    # seq[j-1] connects to seq[i]
                    # seq[i] connects to seq[j+1]
                    
                    n1 = n - overlaps[seq[i-1]][seq[j]] if i > 0 else 0
                    n2 = n - overlaps[seq[j]][seq[i+1]] if i < M-1 else 0
                    n3 = n - overlaps[seq[j-1]][seq[i]] if j > 0 else 0
                    n4 = n - overlaps[seq[i]][seq[j+1]] if j < M-1 else 0
                    
                    delta = (n1 + n2 + n3 + n4) - (c1 + c2 + c3 + c4)
                
                if delta < 0 or random.random() < math.exp(-delta / T):
                    seq[i], seq[j] = seq[j], seq[i]
                    current_energy += delta
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_seq = seq.copy()
                
                T *= cooling
                if T < T_min:
                    T = T_min
            
            # Reconstruct best word
            best_word = list(perms[best_seq[0]])
            for k in range(1, M):
                prev = perms[best_seq[k-1]]
                curr = perms[best_seq[k]]
                ov = overlaps[best_seq[k-1]][best_seq[k]]
                best_word.extend(curr[ov:])
            
            best_word_str = "".join(map(str, best_word))
            
            if len(best_word_str) < baseline_len:
                return best_word_str
                
        except Exception as e:
            pass
            
    return baseline_word
