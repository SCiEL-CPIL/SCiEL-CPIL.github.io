import time
import random
import math

def construct(n):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # 1. Generate the standard superpermutation sequence
    # This mimics the seed algorithm's logic to get the initial sequence of permutations
    # The seed algorithm builds the string, but we need the sequence of permutations
    # to perform TSP local search.
    
    # Standard construction:
    # Start with [1]
    # For m in 2..n:
    #   Find all permutations of length m-1 that appear in the current word.
    #   For each such permutation P, insert m to form P m P.
    #   This effectively defines an ordering of permutations.
    
    # However, it's easier to just generate the standard word and then extract the permutations?
    # No, extracting permutations from the word is O(L*n).
    # It's better to generate the sequence of permutations directly using the recursive logic.
    
    def get_standard_sequence(n):
        if n == 1:
            return [(1,)]
        
        # We need to simulate the "order of appearance" of permutations in the previous word.
        # The seed code does:
        # for i in range(len(word) - m + 2):
        #    window = tuple(word[i:i + m - 1])
        #    if ... window not in seen:
        #       perms.append(window)
        
        # So we need to generate the word for n-1, extract the unique permutations in order,
        # and then expand them.
        
        word = [1]
        seq = [(1,)] # Sequence of permutations for n=1
        
        for m in range(2, n + 1):
            # Extract unique permutations of length m-1 from current word in order of appearance
            prev_perms, seen = [], set()
            for i in range(len(word) - (m - 1) + 1):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    prev_perms.append(window)
            
            # Expand each prev_perm P to P m P
            # The new word is formed by overlapping these blocks.
            # But we just need the sequence of permutations of length m.
            # The seed code constructs the word. The "permutations" in the new word
            # are not simply the expanded blocks; they are the windows of length m in the new word.
            # However, the TSP formulation requires a sequence of ALL n! permutations.
            # Does the standard construction visit all permutations in a specific order?
            # Yes, the windows of length n in the final superpermutation form a sequence of all n! permutations.
            
            # So, for n=7, we can just generate the standard word, then extract all length-n windows.
            # This gives us the initial sequence for TSP.
            
            new = []
            for p in prev_perms:
                block = list(p) + [m] + list(p)
                # Overlap logic
                overlap = 0
                # Find max overlap between end of new and start of block
                max_overlap = min(len(new), len(block))
                for o in range(max_overlap, -1, -1):
                    if o == 0 or new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        
        # Extract all length-n windows from the final word
        final_seq = []
        seen = set()
        for i in range(len(word) - n + 1):
            w = tuple(word[i:i + n])
            if w not in seen:
                seen.add(w)
                final_seq.append(w)
        
        # It should contain all n! permutations
        # len(final_seq) should be n!
        return final_seq

    # Generate initial sequence
    seq = get_standard_sequence(n)
    
    # Calculate initial cost
    def get_overlap(a, b):
        # Max k such that a[-k:] == b[:k]
        max_k = min(len(a), len(b))
        for k in range(max_k, -1, -1):
            if k == 0 or a[-k:] == b[:k]:
                return k
        return 0

    def calc_cost(sequence):
        if not sequence:
            return 0
        cost = n
        for i in range(len(sequence) - 1):
            cost += n - get_overlap(sequence[i], sequence[i + 1])
        return cost

    current_seq = seq
    current_cost = calc_cost(current_seq)
    best_seq = current_seq
    best_cost = current_cost
    
    # Local Search
    # We will use Simulated Annealing
    T = 1.0
    cooling_rate = 0.999
    min_T = 0.01
    
    N = len(current_seq)
    
    def try_move_2opt(seq, i, j):
        # Reverse segment i..j
        if i == j:
            return seq, 0
        new_seq = seq[:i] + seq[i:j+1][::-1] + seq[j+1:]
        return new_seq, 1 # 1 if changed
        
    def try_move_oropt(seq, i, length, new_pos):
        # Move block of 'length' starting at i to new_pos
        if i == new_pos:
            return seq, 0
        block = seq[i:i+length]
        if new_pos > i:
            # Shift left
            new_seq = seq[:i] + seq[i+length:new_pos+length] + block + seq[new_pos+length:]
        else:
            new_seq = seq[:new_pos] + block + seq[new_pos:i] + seq[i+length:]
        return new_seq, 1

    while time.time() < deadline:
        if T < min_T:
            break
            
        # Pick a random move
        move_type = random.random()
        improved = False
        
        if move_type < 0.5:
            # 2-opt
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            new_seq, changed = try_move_2opt(current_seq, i, j)
            if changed:
                new_cost = calc_cost(new_seq)
                delta = new_cost - current_cost
                if delta < 0:
                    current_seq = new_seq
                    current_cost = new_cost
                    improved = True
                else:
                    if random.random() < math.exp(-delta / T):
                        current_seq = new_seq
                        current_cost = new_cost
                        improved = True
        else:
            # Or-opt
            length = random.choice([1, 2, 3])
            if N - length < 1:
                continue
            i = random.randint(0, N - length)
            new_pos = random.randint(0, N - length)
            new_seq, changed = try_move_oropt(current_seq, i, length, new_pos)
            if changed:
                new_cost = calc_cost(new_seq)
                delta = new_cost - current_cost
                if delta < 0:
                    current_seq = new_seq
                    current_cost = new_cost
                    improved = True
                else:
                    if random.random() < math.exp(-delta / T):
                        current_seq = new_seq
                        current_cost = new_cost
                        improved = True
                        
        if current_cost < best_cost:
            best_cost = current_cost
            best_seq = current_seq
            
        T *= cooling_rate
        
    # Construct final string from best_seq
    if not best_seq:
        return "".join(map(str, list(range(1, n+1))))
        
    result = list(best_seq[0])
    for i in range(1, len(best_seq)):
        prev = best_seq[i-1]
        curr = best_seq[i]
        overlap = get_overlap(prev, curr)
        result.extend(curr[overlap:])
        
    return "".join(map(str, result))
