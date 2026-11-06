import time
import sys
import random

def construct():
    start_time = time.time()
    TIME_LIMIT = 55.0
    MAX_VAL = 256
    
    def get_consecutive_sums(seq):
        sums = set()
        n = len(seq)
        for i in range(n):
            total = 0
            for j in range(i, n):
                total += seq[j]
                sums.add(total)
        return sums

    def is_valid(seq):
        if not seq:
            return True
        if seq[0] < 1 or seq[-1] > MAX_VAL:
            return False
        for i in range(len(seq) - 1):
            if seq[i] >= seq[i + 1]:
                return False
        sums = get_consecutive_sums(seq)
        return len(sums) == len(seq) * (len(seq) + 1) // 2

    def greedy_randomized(seed_val=0):
        rng = random.Random(seed_val)
        best = []
        sums = set()
        candidates = list(range(1, MAX_VAL + 1))
        rng.shuffle(candidates)
        for x in candidates:
            tails = [x]
            total = x
            for a in reversed(best):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                best.append(x)
                sums.update(tails)
        best.sort()
        return best

    # Get a good initial solution to set a high bar for pruning
    best_seq = []
    for i in range(500):
        if time.time() > TIME_LIMIT - 20.0:
            break
        seq = greedy_randomized(i)
        if len(seq) > len(best_seq):
            best_seq = seq

    # DFS with bitsets
    # sys.setrecursionlimit(10000)
    
    def dfs(seq, sums_bits, last_val):
        nonlocal best_seq
        if time.time() > TIME_LIMIT:
            return
        
        if len(seq) > len(best_seq):
            best_seq = list(seq)
            
        # Prune: max possible length
        if len(seq) + (MAX_VAL - last_val) <= len(best_seq):
            return

        next_start = last_val + 1
        
        for next_val in range(next_start, MAX_VAL + 1):
            # Check collision
            valid = True
            total = next_val
            if (sums_bits >> total) & 1:
                valid = False
            else:
                for a in reversed(seq):
                    total += a
                    if (sums_bits >> total) & 1:
                        valid = False
                        break
            
            if not valid:
                continue
            
            # Update bitset
            new_sums_bits = sums_bits
            total = next_val
            new_sums_bits |= (1 << total)
            for a in reversed(seq):
                total += a
                new_sums_bits |= (1 << total)
                
            seq.append(next_val)
            dfs(seq, new_sums_bits, next_val)
            seq.pop()

    dfs([], 0, 0)
    
    return best_seq
