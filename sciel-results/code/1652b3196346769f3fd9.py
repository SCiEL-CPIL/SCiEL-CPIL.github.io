import time
import random
import math
import bisect

def _score(S_bits):
    """
    S_bits is a bitset integer where bit i is 1 if i is in S.
    Returns count of distinct d > 0 such that a, a+d, a+2d in S.
    """
    count = 0
    for d in range(1, 256):
        if (S_bits & (S_bits >> d) & (S_bits >> (2 * d))) != 0:
            count += 1
    return count

def _compute_diff_interactions():
    """
    Precompute which pairs of differences (d1, d2) can be simultaneously
    realized by a set of 32 elements. We build a graph where vertices are
    differences d in [1, 255], and edge (d1, d2) exists if there's a set
    of 32 elements that realizes both d1 and d2.
    
    For efficiency, we precompute for each d which positions a, a+d, a+2d
    can exist, and then find high-interaction difference clusters.
    """
    # For each d, find all valid (a, a+d, a+2d) triples within [1, 512]
    diff_triples = {}
    for d in range(1, 256):
        triples = []
        for a in range(1, 512 - 2 * d + 1):
            triples.append((a, a + d, a + 2 * d))
        diff_triples[d] = triples
    
    # Build interaction graph: edge between d1 and d2 if they share at least
    # one common element position that could be part of both progressions
    # This is a heuristic - we'll use it to guide search toward dense clusters
    return diff_triples

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    N = 32
    MAXV = 512
    
    best_score = -1
    best_list = []
    
    # Precompute difference triple information for guided search
    diff_triples = _compute_diff_interactions()
    
    # Strategy: Start with sets that realize many small differences (dense clusters)
    # then use simulated annealing with graph-aware moves
    
    while time.time() < deadline:
        # Initialize with a structured set biased toward small differences
        # Use a base pattern that realizes many small d values
        base = random.randint(1, 50)
        current_list = []
        
        # Add elements in a way that creates many 3-term progressions
        # Start with a dense block
        for i in range(N):
            # Mix of close elements and some spread
            if i < 20:
                # Dense cluster
                val = base + random.randint(0, 30)
            else:
                # Some spread
                val = base + random.randint(0, 200)
            if val not in current_list and 1 <= val <= MAXV:
                current_list.append(val)
        
        # Fill remaining with random
        while len(current_list) < N:
            val = random.randint(1, MAXV)
            if val not in current_list:
                current_list.append(val)
        
        current_list.sort()
        current_bits = 0
        for x in current_list:
            current_bits |= (1 << x)
        
        current_score = _score(current_bits)
        
        if current_score > best_score:
            best_score = current_score
            best_list = current_list.copy()
        
        T = 10.0
        T_min = 0.01
        cooling = 0.999
        
        local_deadline = start_time + min(8.0, deadline - start_time)
        
        while time.time() < local_deadline:
            # Choose move type with graph-aware bias
            move_type = random.random()
            
            if move_type < 0.7:
                # 1-swap biased toward values that create small differences
                x_idx = random.randint(0, N - 1)
                x = current_list[x_idx]
                
                # Bias new value toward creating small-d progressions
                # Pick a value near existing elements
                if random.random() < 0.5:
                    # Near an existing element
                    ref = random.choice(current_list)
                    offset = random.choice([-3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8])
                    y = ref + offset
                    if y < 1 or y > MAXV or ((current_bits >> y) & 1):
                        y = random.randint(1, MAXV)
                        while ((current_bits >> y) & 1):
                            y = random.randint(1, MAXV)
                else:
                    while True:
                        y = random.randint(1, MAXV)
                        if not ((current_bits >> y) & 1):
                            break
                
                new_bits = current_bits ^ (1 << x) ^ (1 << y)
                
                new_list = current_list[:x_idx] + current_list[x_idx+1:]
                ins_idx = bisect.bisect_left(new_list, y)
                new_list.insert(ins_idx, y)
                
            else:
                # 2-swap
                idxs = random.sample(range(N), 2)
                x1 = current_list[idxs[0]]
                x2 = current_list[idxs[1]]
                
                y1 = y2 = 0
                while y1 == 0 or y1 == y2 or ((current_bits >> y1) & 1):
                    y1 = random.randint(1, MAXV)
                while y2 == 0 or y1 == y2 or ((current_bits >> y2) & 1):
                    y2 = random.randint(1, MAXV)
                
                new_bits = current_bits ^ (1 << x1) ^ (1 << x2) ^ (1 << y1) ^ (1 << y2)
                
                if idxs[0] > idxs[1]:
                    idxs[0], idxs[1] = idxs[1], idxs[0]
                
                new_list = current_list[:idxs[0]] + current_list[idxs[0]+1:idxs[1]] + current_list[idxs[1]+1:]
                
                bisect.insort(new_list, y1)
                bisect.insort(new_list, y2)
            
            new_score = _score(new_bits)
            
            delta = new_score - current_score
            
            if delta > 0 or (T > 0 and random.random() < math.exp(delta / T)):
                current_list = new_list
                current_bits = new_bits
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_list = current_list.copy()
            
            T *= cooling
            if T < T_min:
                T = T_min

    return best_list
