import time
import random
from collections import deque
import sys

N = 12
FULL_MASK = (1 << N) - 1
DEADLINE = 85.0

def get_subset_image(mask, trans):
    """Compute image of a subset under a transition."""
    img = 0
    while mask:
        lsb = mask & -mask
        i = lsb.bit_length() - 1
        img |= 1 << trans[i]
        mask ^= lsb
    return img

def calc_reset(trans_a, trans_b):
    """BFS on subset automaton to find shortest reset word."""
    size = 1 << N
    visited = [False] * size
    dist = [0] * size
    
    q = deque()
    q.append(FULL_MASK)
    visited[FULL_MASK] = True
    
    while q:
        cur = q.popleft()
        if cur & (cur - 1) == 0:
            return dist[cur]
        
        d = dist[cur] + 1
        
        nxt_a = get_subset_image(cur, trans_a)
        if not visited[nxt_a]:
            visited[nxt_a] = True
            dist[nxt_a] = d
            q.append(nxt_a)
            
        nxt_b = get_subset_image(cur, trans_b)
        if not visited[nxt_b]:
            visited[nxt_b] = True
            dist[nxt_b] = d
            q.append(nxt_b)
            
    return 0

def gen_tight_candidate():
    """Generate a candidate based on known tight structures."""
    # Try a variation of the Černý automaton
    a = [(i + 1) % N for i in range(N)]
    b = [0] * N
    # Specific pattern
    for i in range(N):
        if i == 0:
            b[i] = 0
        elif i == N - 1:
            b[i] = 0
        else:
            b[i] = (i + 1) % N
    return a, b

def gen_ping_pong(k, m):
    """Generate a ping-pong automaton."""
    n = k + m
    if n != N:
        return None, None
    a = [0] * N
    b = [0] * N
    for i in range(k):
        a[i] = (i + 1) % k
        b[i] = (i - 1) % k
    for i in range(m):
        a[k + i] = k + (i + 1) % m
        b[k + i] = k + (i - 1) % m
    # Connect the two cycles
    a[k - 1] = k
    b[k - 1] = k + m - 1
    a[k + m - 1] = k
    b[k + m - 1] = k
    return a, b

def local_search(a, b, deadline):
    """Simulated annealing local search."""
    cur_a = a[:]
    cur_b = b[:]
    cur_score = calc_reset(cur_a, cur_b)
    
    if cur_score == 0:
        return None, None, 0
        
    best_a = cur_a[:]
    best_b = cur_b[:]
    best_score = cur_score
    
    temp = 1.0
    cooling = 0.99
    
    while time.time() < deadline:
        if temp < 0.01:
            break
            
        # Mutation
        if random.random() < 0.5:
            idx = random.randint(0, N - 1)
            old = cur_a[idx]
            cur_a[idx] = random.randint(0, N - 1)
        else:
            idx = random.randint(0, N - 1)
            old = cur_b[idx]
            cur_b[idx] = random.randint(0, N - 1)
            
        new_score = calc_reset(cur_a, cur_b)
        
        if new_score == 0:
            # Revert
            if random.random() < 0.5:
                cur_a[idx] = old
            else:
                cur_b[idx] = old
            continue
            
        if new_score > best_score:
            best_score = new_score
            best_a = cur_a[:]
            best_b = cur_b[:]
            
        # Acceptance criterion
        if new_score >= cur_score:
            cur_score = new_score
        else:
            delta = new_score - cur_score
            if random.random() < pow(2.71828, delta / temp):
                cur_score = new_score
            else:
                if random.random() < 0.5:
                    cur_a[idx] = old
                else:
                    cur_b[idx] = old
                    
        temp *= cooling
        
    return best_a, best_b, best_score

def construct():
    start = time.time()
    deadline = start + DEADLINE
    
    random.seed(42)
    
    best_a = None
    best_b = None
    best_score = 0
    
    # Phase 1: Structured candidates
    candidates = []
    
    # Ping-pong variants
    for k in range(1, N):
        a, b = gen_ping_pong(k, N - k)
        if a:
            candidates.append((a, b))
            
    # Tight variant
    a, b = gen_tight_candidate()
    candidates.append((a, b))
    
    # Random starts
    for _ in range(50):
        a = [random.randint(0, N - 1) for _ in range(N)]
        b = [random.randint(0, N - 1) for _ in range(N)]
        candidates.append((a, b))
        
    # Evaluate and refine
    for a_c, b_c in candidates:
        if time.time() > deadline - 5.0:
            break
            
        score = calc_reset(a_c, b_c)
        if score > best_score:
            best_score = score
            best_a = a_c[:]
            best_b = b_c[:]
            
        # Refine promising ones
        if score >= best_score - 10 and time.time() < deadline - 10.0:
            ra, rb, rs = local_search(a_c, b_c, deadline - 5.0)
            if rs > best_score:
                best_score = rs
                best_a = ra
                best_b = rb
                
    # Fallback
    if best_score == 0:
        a = [(i + 1) % N for i in range(N)]
        b = [0] * N
        return [a, b]
        
    return [best_a, best_b]

if __name__ == "__main__":
    a, b = construct()
    print(f"Score: {calc_reset(a, b)}")
