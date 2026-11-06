"""Improved program using Simulated Annealing / Local Search."""
import random
import time
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    timeout = 58.0 # Leave 2s buffer
    
    def get_sums(seq):
        sums = set()
        n = len(seq)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                sums.add(s)
        return sums

    def is_valid(seq):
        # Check distinct sums
        sums = get_sums(seq)
        return len(sums) == len(seq) * (len(seq) + 1) // 2

    def greedy_forward():
        values = []
        sums = set()
        for x in range(1, 257):
            tails = []
            total = x
            tails.append(total)
            for a in reversed(values):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                values.append(x)
                sums.update(tails)
        return values

    def greedy_reverse():
        values = []
        sums = set()
        for x in range(256, 0, -1):
            tails = []
            total = x
            tails.append(total)
            for a in reversed(values):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                values.append(x)
                sums.update(tails)
        return sorted(values)

    # Initialize with best greedy
    best_seq = greedy_forward()
    best_len = len(best_seq)
    current_seq = list(best_seq)
    
    # Also try reverse
    rev_seq = greedy_reverse()
    if len(rev_seq) > best_len:
        best_seq = rev_seq
        best_len = len(best_seq)
        current_seq = list(best_seq)

    def try_insert(seq, x, pos):
        new_seq = seq[:pos] + [x] + seq[pos:]
        if is_valid(new_seq):
            return new_seq
        return None

    def try_swap(seq, i, j):
        if i == j: return seq
        new_seq = seq[:]
        new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
        if is_valid(new_seq):
            return new_seq
        return None

    # Local Search Loop
    stuck_counter = 0
    temperature = 1.0
    cooling = 0.999
    
    while time.time() - start_time < timeout:
        n = len(current_seq)
        
        # 1. Attempt Insertion (Primary Goal)
        # Sample a few random candidates for (x, pos)
        used = set(current_seq)
        available = [x for x in range(1, 257) if x not in used]
        
        if available:
            # Try a few random insertions
            attempts = min(5, len(available) * (n + 1))
            found = False
            for _ in range(attempts):
                x = random.choice(available)
                pos = random.randint(0, n)
                new_seq = try_insert(current_seq, x, pos)
                if new_seq:
                    current_seq = new_seq
                    if len(current_seq) > best_len:
                        best_seq = current_seq[:]
                        best_len = len(current_seq)
                    stuck_counter = 0
                    found = True
                    break
            
            if found:
                continue

        # 2. If no insertion, try to modify structure
        # Options: Delete, Swap, or Random Restart
        
        if n > 0:
            # Try Swap
            if n >= 2:
                i, j = random.sample(range(n), 2)
                new_seq = try_swap(current_seq, i, j)
                if new_seq:
                    current_seq = new_seq
                    stuck_counter = 0
                    continue
            
            # Try Delete (to unblock)
            # Only delete if we are stuck or to explore
            if stuck_counter > 10 or random.random() < 0.1:
                idx = random.randint(0, n - 1)
                current_seq = current_seq[:idx] + current_seq[idx+1:]
                stuck_counter += 1
                continue
        
        # If we are very stuck, random restart from a greedy seed
        stuck_counter += 1
        if stuck_counter > 100:
            # Random restart
            if random.random() < 0.5:
                current_seq = greedy_forward()
            else:
                current_seq = greedy_reverse()
            # Perturb the restart
            for _ in range(5):
                if not current_seq: break
                idx = random.randint(0, len(current_seq)-1)
                current_seq = current_seq[:idx] + current_seq[idx+1:]
            stuck_counter = 0
            temperature *= cooling

    return best_seq
# EVOLVE-BLOCK-END
