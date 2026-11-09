import time
import sys
import random
import heapq
from itertools import permutations
from math import factorial

def construct(n):
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin
    
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    
    # Precompute all permutations and their encodings
    all_perms_list = list(permutations(range(1, n + 1)))
    num_perms = len(all_perms_list)
    
    # Map each permutation to an index
    perm_to_idx = {}
    for idx, p in enumerate(all_perms_list):
        perm_to_idx[p] = idx
    
    # Build adjacency: which permutations can follow which
    # Two permutations can be adjacent if the last (n-1) of first = first (n-1) of second
    # For n=7, this is a sparse graph
    adj = [[] for _ in range(num_perms)]
    for idx, p in enumerate(all_perms_list):
        suffix = p[1:]  # last n-1 elements
        for jdx, q in enumerate(all_perms_list):
            if q[:-1] == suffix:
                adj[idx].append(jdx)
    
    # For each permutation, find all permutations that can follow it
    # This is the de Bruijn graph structure
    # Actually, for superpermutations, we need a different structure
    # Let's use the classic approach: build a word that contains all permutations
    
    # Classic superpermutation construction
    def classic_superperm(n):
        if n == 1:
            return [1]
        prev = classic_superperm(n - 1)
        perms = []
        seen = set()
        for i in range(len(prev) - (n - 2)):
            window = tuple(prev[i:i + n - 1])
            if len(set(window)) == n - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [n] + list(p)
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        return new
    
    word = classic_superperm(n)
    
    # Precompute powers for fast window encoding
    n_powers = [1] * (n + 1)
    for i in range(1, n + 1):
        n_powers[i] = n_powers[i-1] * n
    n_pow_n_minus_1 = n_powers[n-1]
    
    def encode_window_fast(w):
        code = 0
        for x in w:
            code = code * n + (x - 1)
        return code
    
    # Precompute all valid permutation encodings
    all_perms_set = set()
    for p in permutations(range(1, n + 1)):
        code = 0
        for x in p:
            code = code * n + (x - 1)
        all_perms_set.add(code)
    
    def build_window_codes(word, n):
        L = len(word)
        num_windows = L - n + 1
        window_codes = [0] * num_windows
        if num_windows <= 0:
            return window_codes
        code = 0
        for j in range(n):
            code = code * n + (word[j] - 1)
        window_codes[0] = code
        for i in range(1, num_windows):
            code = (code - (word[i-1] - 1) * n_pow_n_minus_1) * n + (word[i+n-1] - 1)
            window_codes[i] = code
        return window_codes
    
    # Build initial window codes
    window_codes = build_window_codes(word, n)
    
    # Count occurrences of each permutation
    counts = {}
    for code in window_codes:
        if code in all_perms_set:
            counts[code] = counts.get(code, 0) + 1
    
    # Greedy pruning
    def try_prune(word, window_codes, counts, n, deadline, direction=1):
        L = len(word)
        removed_any = False
        positions = list(range(L))
        if direction == -1:
            positions.reverse()
        
        for i in positions:
            if time.time() >= deadline:
                break
            L = len(word)
            if L < n:
                break
            if i >= L:
                continue
            
            start_min = max(0, i - n + 1)
            start_max = min(i, L - n)
            
            safe = True
            for s in range(start_min, start_max + 1):
                code = window_codes[s]
                if code in all_perms_set and counts.get(code, 0) == 1:
                    safe = False
                    break
            
            if not safe:
                continue
            
            new_codes = {}
            for s in range(start_min, start_max + 1):
                if s < i:
                    w_new = word[s:i] + word[i+1:s+n+1]
                else:
                    w_new = word[i+1:i+n+1]
                code = encode_window_fast(w_new)
                new_codes[s] = code
            
            for s in range(start_min, start_max + 1):
                code = window_codes[s]
                if code in all_perms_set:
                    counts[code] -= 1
                    if counts[code] <= 0:
                        del counts[code]
            
            for s, code in new_codes.items():
                if code in all_perms_set:
                    counts[code] = counts.get(code, 0) + 1
            
            word.pop(i)
            window_codes = build_window_codes(word, n)
            removed_any = True
        
        return word, window_codes, counts, removed_any
    
    # Multiple passes in both directions
    for _ in range(50):
        if time.time() >= deadline:
            break
        word, window_codes, counts, removed_any = try_prune(word, window_codes, counts, n, deadline, direction=1)
        if not removed_any:
            break
        if time.time() >= deadline:
            break
        word, window_codes, counts, removed_any = try_prune(word, window_codes, counts, n, deadline, direction=-1)
        if not removed_any:
            break
    
    # Genetic Algorithm Phase
    # Represent a solution as a sequence of permutation indices
    # The word is built by overlapping consecutive permutations
    
    def build_word_from_seq(seq, n):
        """Build word from sequence of permutation indices"""
        if not seq:
            return []
        word = list(all_perms_list[seq[0]])
        for idx in seq[1:]:
            p = all_perms_list[idx]
            # Find maximum overlap between end of word and start of p
            max_ov = min(len(word), n - 1)
            ov = 0
            for o in range(max_ov, -1, -1):
                if o == 0 or word[len(word)-o:] == list(p[:o]):
                    ov = o
                    break
            word.extend(p[ov:])
        return word
    
    def is_valid_superperm(word, n, all_perms_set):
        """Check if word contains all permutations"""
        if len(word) < n:
            return False
        found = set()
        code = 0
        for j in range(n):
            code = code * n + (word[j] - 1)
        if code in all_perms_set:
            found.add(code)
        for i in range(1, len(word) - n + 1):
            code = (code - (word[i-1] - 1) * n_pow_n_minus_1) * n + (word[i+n-1] - 1)
            if code in all_perms_set:
                found.add(code)
        return len(found) == num_perms
    
    def get_fitness(word, n, all_perms_set):
        """Fitness = number of unique permutations found (higher is better)"""
        if len(word) < n:
            return 0
        found = set()
        code = 0
        for j in range(n):
            code = code * n + (word[j] - 1)
        if code in all_perms_set:
            found.add(code)
        for i in range(1, len(word) - n + 1):
            code = (code - (word[i-1] - 1) * n_pow_n_minus_1) * n + (word[i+n-1] - 1)
            if code in all_perms_set:
                found.add(code)
        return len(found)
    
    # Create initial population from classic superpermutation
    # Extract permutation sequence from classic word
    def extract_perm_seq(word, n, all_perms_set):
        """Extract sequence of permutations from word"""
        seq = []
        seen_positions = set()
        for i in range(len(word) - n + 1):
            window = tuple(word[i:i+n])
            code = 0
            for x in window:
                code = code * n + (x - 1)
            if code in all_perms_set and i not in seen_positions:
                # Check if this is a new permutation not already covered
                idx = perm_to_idx.get(window, -1)
                if idx >= 0:
                    seq.append(idx)
                    seen_positions.add(i)
        return seq
    
    # Initial sequence from classic superpermutation
    init_seq = extract_perm_seq(word, n, all_perms_set)
    
    # Population for GA
    pop_size = 20
    population = [init_seq[:]]
    for _ in range(pop_size - 1):
        # Create variations
        mut_seq = init_seq[:]
        for _ in range(random.randint(1, 3)):
            if mut_seq:
                pos = random.randint(0, len(mut_seq) - 1)
                # Swap with adjacent
                if pos > 0 and random.random() < 0.5:
                    mut_seq[pos-1], mut_seq[pos] = mut_seq[pos], mut_seq[pos-1]
                # Or replace with a compatible permutation
                else:
                    if pos < len(mut_seq) - 1:
                        next_idx = mut_seq[pos+1]
                        next_perm = all_perms_list[next_idx]
                        # Find a permutation that ends with same prefix as next_perm starts
                        candidates = [j for j in range(num_perms) if all_perms_list[j][1:] == next_perm[:-1]]
                        if candidates:
                            mut_seq[pos] = random.choice(candidates)
        population.append(mut_seq)
    
    best_seq = init_seq[:]
    best_word = build_word_from_seq(best_seq, n)
    best_fitness = get_fitness(best_word, n, all_perms_set)
    
    # GA evolution
    gen = 0
    max_gens = 200
    while gen < max_gens and time.time() < deadline:
        gen += 1
        
        # Evaluate fitness
        fitnesses = []
        for seq in population:
            w = build_word_from_seq(seq, n)
            f = get_fitness(w, n, all_perms_set)
            fitnesses.append(f)
            if f > best_fitness:
                best_fitness = f
                best_seq = seq[:]
                best_word = w[:]
        
        # Selection: tournament
        new_population = []
        for _ in range(pop_size):
            # Tournament of 3
            cands = random.sample(range(pop_size), 3)
            winner = max(cands, key=lambda i: fitnesses[i])
            new_population.append(population[winner][:])
        
        # Crossover
        for i in range(0, len(new_population) - 1, 2):
            if random.random() < 0.8:
                # Uniform crossover
                p1, p2 = new_population[i], new_population[i+1]
                if p1 and p2:
                    cross_point = random.randint(1, min(len(p1), len(p2)) - 1)
                    c1 = p1[:cross_point] + p2[cross_point:]
                    c2 = p2[:cross_point] + p1[cross_point:]
                    new_population[i] = c1
                    new_population[i+1] = c2
        
        # Mutation
        for i in range(len(new_population)):
            seq = new_population[i]
            if not seq:
                continue
            # Multiple mutations
            num_muts = random.randint(1, 4)
            for _ in range(num_muts):
                if not seq:
                    break
                mut_type = random.random()
                pos = random.randint(0, len(seq) - 1)
                if mut_type < 0.3:
                    # Swap adjacent
                    if pos > 0:
                        seq[pos-1], seq[pos] = seq[pos], seq[pos-1]
                    elif pos < len(seq) - 1:
                        seq[pos], seq[pos+1] = seq[pos+1], seq[pos]
                elif mut_type < 0.6:
                    # Replace with compatible permutation
                    if pos < len(seq) - 1:
                        next_idx = seq[pos+1]
                        next_perm = all_perms_list[next_idx]
                        candidates = [j for j in range(num_perms) if all_perms_list[j][1:] == next_perm[:-1]]
                        if candidates:
                            seq[pos] = random.choice(candidates)
                    if pos > 0:
                        prev_idx = seq[pos-1]
                        prev_perm = all_perms_list[prev_idx]
                        candidates = [j for j in range(num_perms) if prev_perm[1:] == all_perms_list[j][:-1]]
                        if candidates:
                            seq[pos] = random.choice(candidates)
                elif mut_type < 0.8:
                    # Insert a permutation
                    if pos < len(seq):
                        cur_idx = seq[pos]
                        cur_perm = all_perms_list[cur_idx]
                        candidates = [j for j in range(num_perms) if cur_perm[1:] == all_perms_list[j][:-1]]
                        if candidates:
                            seq.insert(pos, random.choice(candidates))
                else:
                    # Delete a permutation
                    if len(seq) > 1:
                        seq.pop(pos)
        
        population = new_population
        
        # Elitism: keep best
        if best_seq:
            population[0] = best_seq[:]
    
    # Final: use best word found
    final_word = best_word if len(best_word) >= n else word
    
    # One more round of greedy pruning on the best word
    if time.time() < deadline:
        window_codes = build_window_codes(final_word, n)
        counts = {}
        for code in window_codes:
            if code in all_perms_set:
                counts[code] = counts.get(code, 0) + 1
        
        for _ in range(30):
            if time.time() >= deadline:
                break
            final_word, window_codes, counts, removed_any = try_prune(final_word, window_codes, counts, n, deadline, direction=1)
            if not removed_any:
                break
            if time.time() >= deadline:
                break
            final_word, window_codes, counts, removed_any = try_prune(final_word, window_codes, counts, n, deadline, direction=-1)
            if not removed_any:
                break
    
    return "".join(map(str, final_word))

if __name__ == "__main__":
    res = construct(7)
    print(len(res))
