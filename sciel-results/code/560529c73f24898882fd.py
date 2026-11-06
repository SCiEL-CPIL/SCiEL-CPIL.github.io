"""Two-colourings of [1,256] with few monochromatic 4-term progressions."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

N = 256
K = 4

def count_mono_4ap_fast(coloring):
    """Count monochromatic 4-term APs in a 0/1 coloring of 1..256."""
    c = np.array(coloring, dtype=np.int8)
    count = 0
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        if max_a < 1:
            break
        seq1 = c[:max_a]
        seq2 = c[d:d + max_a]
        seq3 = c[2 * d:2 * d + max_a]
        seq4 = c[3 * d:3 * d + max_a]
        mono = (seq1 == seq2) & (seq2 == seq3) & (seq3 == seq4)
        count += np.sum(mono)
    return count

def flip_delta(coloring, pos):
    """Compute the change in mono-4AP count if we flip coloring[pos].
    Returns the delta (new_score - old_score) without actually flipping."""
    c = np.array(coloring, dtype=np.int8)
    old_val = c[pos]
    new_val = 1 - old_val
    delta = 0
    
    # A 4-AP a, a+d, a+2d, a+3d contains pos if pos == a, a+d, a+2d, or a+3d
    # For each such AP, check if it was mono before and if it becomes mono after flip
    
    # Case 1: pos = a (first element)
    # a = pos, need a+3d <= N => pos + 3d <= N => d <= (N - pos) // 3
    for d in range(1, (N - pos) // 3 + 1):
        a = pos
        if a + 3 * d > N:
            break
        idx = [a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1]
        # Check if all four are same color
        if c[idx[0]] == c[idx[1]] == c[idx[2]] == c[idx[3]]:
            # Was mono, now not mono (since we're flipping pos)
            delta -= 1
        # Check if after flip it becomes mono
        # After flip, c[pos] = new_val
        if new_val == c[idx[1]] == c[idx[2]] == c[idx[3]]:
            # Becomes mono
            delta += 1
    
    # Case 2: pos = a+d (second element)
    # a+d = pos => a = pos - d, need a >= 1 => d <= pos - 1
    # and a+3d = pos + 2d <= N => d <= (N - pos) // 2
    for d in range(1, min(pos, (N - pos) // 2) + 1):
        a = pos - d
        if a < 1:
            break
        idx = [a - 1, pos, a + 2 * d - 1, a + 3 * d - 1]
        if c[idx[0]] == c[idx[1]] == c[idx[2]] == c[idx[3]]:
            delta -= 1
        if c[idx[0]] == new_val == c[idx[2]] == c[idx[3]]:
            delta += 1
    
    # Case 3: pos = a+2d (third element)
    # a+2d = pos => a = pos - 2d, need a >= 1 => d <= (pos-1) // 2
    # a+3d = pos + d <= N => d <= N - pos
    for d in range(1, min((pos - 1) // 2, N - pos) + 1):
        a = pos - 2 * d
        if a < 1:
            break
        idx = [a - 1, a + d - 1, pos, a + 3 * d - 1]
        if c[idx[0]] == c[idx[1]] == c[idx[2]] == c[idx[3]]:
            delta -= 1
        if c[idx[0]] == c[idx[1]] == new_val == c[idx[3]]:
            delta += 1
    
    # Case 4: pos = a+3d (fourth element)
    # a+3d = pos => a = pos - 3d, need a >= 1 => d <= (pos-1) // 3
    for d in range(1, (pos - 1) // 3 + 1):
        a = pos - 3 * d
        if a < 1:
            break
        idx = [a - 1, a + d - 1, a + 2 * d - 1, pos]
        if c[idx[0]] == c[idx[1]] == c[idx[2]] == c[idx[3]]:
            delta -= 1
        if c[idx[0]] == c[idx[1]] == c[idx[2]] == new_val:
            delta += 1
    
    return delta

def local_search(coloring, deadline, max_iters=5000):
    """Hill-climbing local search with delta evaluation."""
    c = coloring[:]
    score = count_mono_4ap_fast(c)
    best_c = c[:]
    best_score = score
    
    it = 0
    while it < max_iters and time.time() < deadline:
        improved = False
        best_pos = -1
        best_delta = 0
        
        # Try each position, pick the one with most negative delta
        for pos in range(N):
            d = flip_delta(c, pos)
            if d < best_delta:
                best_delta = d
                best_pos = pos
        
        if best_pos >= 0 and best_delta < 0:
            c[best_pos] = 1 - c[best_pos]
            score += best_delta
            improved = True
            if score < best_score:
                best_score = score
                best_c = c[:]
        
        if not improved:
            # Try random flips to escape local minima
            for _ in range(10):
                if time.time() > deadline:
                    break
                pos = random.randint(0, N - 1)
                c[pos] = 1 - c[pos]
                score += flip_delta(c, pos)
                # Wait, we just flipped, so we need to recompute
                # Actually let's just recompute
                score = count_mono_4ap_fast(c)
                if score < best_score:
                    best_score = score
                    best_c = c[:]
            break
        
        it += 1
    
    return best_c, best_score

def construct():
    deadline = time.time() + 130  # Safety margin
    
    # Generate diverse starting points
    starts = []
    
    # Random
    for _ in range(30):
        starts.append([random.randint(0, 1) for _ in range(N)])
    
    # Block patterns
    for bs in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 25, 32, 40, 50, 64, 80, 100, 128]:
        if bs <= N:
            starts.append([(i // bs) % 2 for i in range(N)])
    
    # Mod patterns
    for mod in [2, 3, 4, 5, 6, 7, 8, 10, 12, 16]:
        starts.append([1 if (i % mod) < mod // 2 else 0 for i in range(N)])
    
    # Alternating
    starts.append([i % 2 for i in range(N)])
    
    # Alternating with noise
    for noise in [0.1, 0.2, 0.3]:
        base = [i % 2 for i in range(N)]
        starts.append([b if random.random() > noise else 1 - b for b in base])
    
    # Greedy construction
    greedy = [-1] * (N + 1)
    for j in range(1, N + 1):
        best_color = 0
        best_cnt = float('inf')
        for color in (0, 1):
            cnt = 0
            for d in range(1, (j // 3) + 1):
                a = j - 3 * d
                if a < 1:
                    break
                if greedy[a] != -1 and greedy[a + d] != -1 and greedy[a + 2 * d] != -1:
                    if greedy[a] == color and greedy[a + d] == color and greedy[a + 2 * d] == color:
                        cnt += 1
            if cnt < best_cnt:
                best_cnt = cnt
                best_color = color
        greedy[j] = best_color
    starts.append(greedy[1:])
    
    best_coloring = None
    best_score = float('inf')
    
    # Run local search from each starting point
    for start in starts:
        if time.time() > deadline - 20:
            break
        c, s = local_search(start, deadline - 20, max_iters=2000)
        if s < best_score:
            best_score = s
            best_coloring = c
    
    # Simulated annealing from best
    if time.time() < deadline - 15 and best_coloring is not None:
        current = best_coloring[:]
        current_score = best_score
        best = best_coloring[:]
        T = 5.0
        alpha = 0.9995
        steps = 0
        while time.time() < deadline - 5 and T > 0.01:
            pos = random.randint(0, N - 1)
            current[pos] = 1 - current[pos]
            new_score = count_mono_4ap_fast(current)
            delta = new_score - current_score
            if delta <= 0 or random.random() < np.exp(-delta / max(T, 1e-10)):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best = current[:]
            else:
                current[pos] = 1 - current[pos]
            T *= alpha
            steps += 1
            # Occasional local search polish
            if steps % 500 == 0 and time.time() < deadline - 10:
                c, s = local_search(current, deadline - 10, max_iters=200)
                if s < best_score:
                    best_score = s
                    best = c
                    current = c[:]
    
    # Final polish
    if time.time() < deadline - 5 and best_coloring is not None:
        best_coloring, best_score = local_search(best_coloring, deadline - 5, max_iters=1000)
    
    return best_coloring
# EVOLVE-BLOCK-END
