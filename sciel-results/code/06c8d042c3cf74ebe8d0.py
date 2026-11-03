import numpy as np
import itertools
import time
import random
from collections import defaultdict
from collections import deque

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 112  # 8s buffer for safety
    
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters: simple greedy
        N = q ** n
        def word_to_idx(word):
            idx = 0
            for i in range(n):
                idx = idx * q + word[i]
            return idx
        def idx_to_word(idx):
            word = [0] * n
            for i in range(n-1, -1, -1):
                word[i] = idx % q
                idx //= q
            return word
        all_words = [idx_to_word(i) for i in range(N)]
        neighbor_masks = [0] * N
        for c_idx in range(N):
            c_word = all_words[c_idx]
            mask = 1 << c_idx
            if radius == 1:
                for i in range(n):
                    orig = c_word[i]
                    for new_val in range(q):
                        if new_val != orig:
                            new_word = c_word.copy()
                            new_word[i] = new_val
                            w_idx = word_to_idx(new_word)
                            mask |= (1 << w_idx)
            neighbor_masks[c_idx] = mask
        uncovered = (1 << N) - 1
        code = []
        while uncovered and time.time() < deadline:
            best_c = -1
            best_count = -1
            for c in range(N):
                count = (uncovered & neighbor_masks[c]).bit_count()
                if count > best_count:
                    best_count = count
                    best_c = c
            if best_c == -1:
                break
            code.append(best_c)
            uncovered &= ~neighbor_masks[best_c]
        return [all_words[i] for i in code]
    
    # Main path: q=3, n=8, radius=1
    N = 3 ** 8  # 6561
    
    # Precompute neighbor lists as numpy arrays
    neighbor_lists = []
    for c in range(N):
        word = []
        temp = c
        for i in range(8):
            word.append(temp % 3)
            temp //= 3
        word = word[::-1]  # word[0] is most significant
        
        neighbors = [c]
        for pos in range(8):
            orig = word[pos]
            for delta in [1, 2]:
                new_val = (orig + delta) % 3
                new_word = word[:]
                new_word[pos] = new_val
                idx = 0
                for j in range(8):
                    idx = idx * 3 + new_word[j]
                neighbors.append(idx)
        neighbor_lists.append(neighbors)
    
    neighbor_arrays = [np.array(nl, dtype=np.int32) for nl in neighbor_lists]
    
    def time_left():
        return deadline - time.time()
    
    def build_cover_counts(code):
        counts = np.zeros(N, dtype=np.int32)
        for c in code:
            counts[neighbor_arrays[c]] += 1
        return counts
    
    def idx_to_word(idx):
        word = [0] * 8
        for i in range(7, -1, -1):
            word[i] = idx % 3
            idx //= 3
        return word
    
    def word_to_idx(word):
        idx = 0
        for i in range(8):
            idx = idx * 3 + word[i]
        return idx
    
    def algebraic_init():
        """
        Deterministic structured initialization using the tetracode product structure.
        
        GF(3)^8 = GF(3)^4 x GF(3)^4.
        The tetracode is a [4,2,3] code over GF(3) with 9 codewords.
        We use a product-like structure: for each pair of tetracode codewords,
        create a codeword in GF(3)^8. This gives 9*9 = 81 codewords initially,
        which we then complete with a criticality-driven greedy.
        
        Actually, let's use a denser structure. The known optimal is 486 = 2 * 3^6 / 2 = 2*729/3 = 486.
        486 = 6 * 81. So we can think of it as 6 cosets of a subgroup of size 81.
        
        Let's use: for each (a,b) where a in GF(3)^4, b in GF(3)^4,
        include (a,b) if sum(a) == sum(b) mod 3. That gives 3^8/3 = 2187... too many.
        
        Better: use a structured seed of ~400-500 well-spread points.
        """
        random.seed(42)
        
        # Use a lattice structure: take points where certain linear constraints hold
        # This gives a structured, well-spread set
        code = []
        code_set = set()
        
        # Generate a structured set using linear constraints
        # For each of the 3^4 = 81 possible first-half values,
        # pick a few second-half values that are well-spread
        for a in range(81):
            a_word = idx_to_word(a)
            a_word = a_word[:4]  # first 4 digits
            
            # Pick 6 second-half values spread across GF(3)^4
            for b_offset in range(6):
                b_word = [(a_word[0] + b_offset) % 3,
                         (a_word[1] + (b_offset * 2) % 3) % 3,
                         (a_word[2] + (b_offset * b_offset) % 3) % 3,
                         (a_word[3] + (b_offset * 3) % 3) % 3]
                
                full_word = a_word + b_word
                idx = word_to_idx(full_word)
                
                if idx not in code_set:
                    code.append(idx)
                    code_set.add(idx)
        
        # Complete with criticality-driven greedy
        counts = build_cover_counts(code)
        
        # Precompute for each word, how many codewords can cover it (criticality)
        # criticality[w] = number of codewords c such that w is in neighbors(c)
        # We want to cover the most critical (hardest to cover) words first
        
        while True:
            if time_left() < 3:
                break
            
            uncovered = np.where(counts == 0)[0]
            if len(uncovered) == 0:
                break
            
            # For each uncovered word, compute its criticality (how few codewords can cover it)
            # Actually, just do standard greedy: pick the codeword that covers the most uncovered
            best_c = -1
            best_count = 0
            
            for c in range(N):
                if c in code_set:
                    continue
                covered = np.sum(counts[neighbor_arrays[c]] == 0)
                if covered > best_count:
                    best_count = covered
                    best_c = c
            
            if best_c == -1 or best_count == 0:
                break
            
            code.append(best_c)
            code_set.add(best_c)
            counts[neighbor_arrays[best_c]] += 1
        
        return code
    
    def greedy_complete(code):
        """Complete a partial covering with standard greedy."""
        code_set = set(code)
        counts = build_cover_counts(code)
        
        while True:
            if time_left() < 2:
                break
            uncovered = np.where(counts == 0)[0]
            if len(uncovered) == 0:
                break
            
            best_c = -1
            best_count = 0
            for c in range(N):
                if c in code_set:
                    continue
                covered = np.sum(counts[neighbor_arrays[c]] == 0)
                if covered > best_count:
                    best_count = covered
                    best_c = c
            
            if best_c == -1:
                break
            
            code.append(best_c)
            code_set.add(best_c)
            counts[neighbor_arrays[best_c]] += 1
        
        return code
    
    def local_search_remove(code):
        """
        Strict local search: repeatedly try to remove codewords.
        A codeword can be removed if no word is uniquely covered by it.
        """
        code = list(code)
        code_set = set(code)
        counts = build_cover_counts(code)
        
        improved = True
        while improved and time_left() > 2:
            improved = False
            for i in range(len(code) - 1, -1, -1):
                c = code[i]
                if np.any(counts[neighbor_arrays[c]] == 1):
                    continue
                # Safe to remove
                code.pop(i)
                code_set.discard(c)
                counts[neighbor_arrays[c]] -= 1
                improved = True
                break
        
        return code
    
    def two_opt_local_search(code):
        """
        2-opt: try removing two codewords and adding one that covers all
        newly-uncovered words. Also try single swaps.
        """
        code = list(code)
        code_set = set(code)
        counts = build_cover_counts(code)
        
        improved = True
        while improved and time_left() > 3:
            improved = False
            
            # Try single removals first
            for i in range(len(code) - 1, -1, -1):
                if time_left() < 2:
                    break
                c = code[i]
                if np.any(counts[neighbor_arrays[c]] == 1):
                    continue
                code.pop(i)
                code_set.discard(c)
                counts[neighbor_arrays[c]] -= 1
                improved = True
                break
            
            if improved:
                continue
            
            # Try single swaps: remove one, add one
            for i in range(len(code)):
                if time_left() < 2:
                    break
                
                c_remove = code[i]
                newly_uncovered = neighbor_arrays[c_remove][counts[neighbor_arrays[c_remove]] == 1]
                
                if len(newly_uncovered) == 0:
                    code.pop(i)
                    code_set.discard(c_remove)
                    counts[neighbor_arrays[c_remove]] -= 1
                    improved = True
                    break
                
                # Find a replacement that covers all newly_uncovered
                best_add = -1
                best_gain = -1
                
                for c_add in range(N):
                    if c_add in code_set:
                        continue
                    if not np.all(np.isin(newly_uncovered, neighbor_arrays[c_add])):
                        continue
                    gain = np.sum(counts[neighbor_arrays[c_add]] == 0)
                    if gain > best_gain:
                        best_gain = gain
                        best_add = c_add
                
                if best_add != -1 and best_gain >= 0:
                    code.pop(i)
                    code_set.discard(c_remove)
                    counts[neighbor_arrays[c_remove]] -= 1
                    code.append(best_add)
                    code_set.add(best_add)
                    counts[neighbor_arrays[best_add]] += 1
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try 2-removal + 1-addition
            for i in range(len(code) - 1):
                if time_left() < 2:
                    break
                for j in range(i + 1, len(code)):
                    if time_left() < 2:
                        break
                    
                    c1 = code[i]
                    c2 = code[j]
                    
                    # Words uniquely covered by c1 or c2
                    newly_uncovered = set()
                    for w in neighbor_arrays[c1]:
                        if counts[w] == 1:
                            newly_uncovered.add(int(w))
                    for w in neighbor_arrays[c2]:
                        if counts[w] == 1:
                            newly_uncovered.add(int(w))
                    
                    if len(newly_uncovered) == 0:
                        # Remove both
                        code.pop(j)
                        code.pop(i)
                        code_set.discard(c1)
                        code_set.discard(c2)
                        counts[neighbor_arrays[c1]] -= 1
                        counts[neighbor_arrays[c2]] -= 1
                        improved = True
                        break
                    
                    if len(newly_uncovered) > 17:  # Can't be covered by one codeword
                        continue
                    
                    # Find a codeword that covers all newly_uncovered
                    best_add = -1
                    best_gain = -1
                    newly_uncovered_arr = np.array(list(newly_uncovered), dtype=np.int32)
                    
                    for c_add in range(N):
                        if c_add in code_set:
                            continue
                        if not np.all(np.isin(newly_uncovered_arr, neighbor_arrays[c_add])):
                            continue
                        gain = np.sum(counts[neighbor_arrays[c_add]] == 0)
                        if gain > best_gain:
                            best_gain = gain
                            best_add = c_add
                    
                    if best_add != -1 and best_gain >= 1:
                        # Remove both, add one
                        code.pop(j)
                        code.pop(i)
                        code_set.discard(c1)
                        code_set.discard(c2)
                        counts[neighbor_arrays[c1]] -= 1
                        counts[neighbor_arrays[c2]] -= 1
                        code.append(best_add)
                        code_set.add(best_add)
                        counts[neighbor_arrays[best_add]] += 1
                        improved = True
                        break
                
                if improved:
                    break
            
            if improved:
                continue
            
            # Try 1-removal + 2-additions (net -1)
            for i in range(len(code)):
                if time_left() < 2:
                    break
                
                c_remove = code[i]
                newly_uncovered = neighbor_arrays[c_remove][counts[neighbor_arrays[c_remove]] == 1]
                
                if len(newly_uncovered) == 0:
                    code.pop(i)
                    code_set.discard(c_remove)
                    counts[neighbor_arrays[c_remove]] -= 1
                    improved = True
                    break
                
                if len(newly_uncovered) > 17:
                    continue
                
                # Find two codewords that together cover all newly_uncovered
                newly_uncovered_set = set(int(w) for w in newly_uncovered)
                best_pair = None
                best_gain = -1
                
                # Sample a subset of candidates for speed
                candidates = [c for c in range(N) if c not in code_set]
                random.shuffle(candidates)
                candidates = candidates[:200]
                
                for c_add1 in candidates:
                    na1 = neighbor_arrays[c_add1]
                    covered_by_1 = set(int(w) for w in na1 if w in newly_uncovered_set)
                    remaining = newly_uncovered_set - covered_by_1
                    
                    if not remaining:
                        gain = np.sum(counts[na1] == 0)
                        if gain > best_gain:
                            best_gain = gain
                            best_pair = (c_add1, -1)
                        continue
                    
                    for c_add2 in candidates:
                        if c_add2 == c_add1:
                            continue
                        na2 = neighbor_arrays[c_add2]
                        covered_by_2 = set(int(w) for w in na2 if w in remaining)
                        if remaining.issubset(covered_by_2):
                            gain = np.sum((counts[na1] == 0) | (counts[na2] == 0))
                            if gain > best_gain:
                                best_gain = gain
                                best_pair = (c_add1, c_add2)
                
                if best_pair is not None and best_gain >= 1:
                    code.pop(i)
                    code_set.discard(c_remove)
                    counts[neighbor_arrays[c_remove]] -= 1
                    for c_add in best_pair:
                        if c_add != -1:
                            code.append(c_add)
                            code_set.add(c_add)
                            counts[neighbor_arrays[c_add]] += 1
                    improved = True
                    break
        
        return code
    
    def genetic_algorithm(code, pop_size=16, generations=40):
        """
        Genetic algorithm on binary genomes (codeword presence).
        Fitness: minimize code size + heavy penalty for uncovered words.
        Uses uniform crossover with coverage-repair and targeted mutation.
        """
        code = list(code)
        code_set = set(code)
        counts = build_cover_counts(code)
        
        # Build initial population: parent + random perturbations
        population = [list(code)]
        for _ in range(pop_size - 1):
            pop = list(code)
            # Remove some random codewords
            n_remove = random.randint(5, 20)
            for _ in range(n_remove):
                if pop:
                    idx = random.randint(0, len(pop) - 1)
                    pop.pop(idx)
            # Add some random codewords
            n_add = random.randint(5, 20)
            for _ in range(n_add):
                c = random.randint(0, N - 1)
                if c not in pop:
                    pop.append(c)
            population.append(pop)
        
        def fitness(pop):
            """Lower is better. Penalize uncovered words heavily."""
            if not pop:
                return N + 1000
            counts_f = np.zeros(N, dtype=np.int32)
            for c in pop:
                counts_f[neighbor_arrays[c]] += 1
            uncovered = int(np.sum(counts_f == 0))
            return len(pop) + 100 * uncovered
        
        best_overall = list(code)
        best_fitness = fitness(code)
        
        for gen in range(generations):
            if time_left() < 5:
                break
            
            # Evaluate fitness
            fits = [fitness(pop) for pop in population]
            
            # Track best
            for i in range(len(population)):
                if fits[i] < best_fitness:
                    best_fitness = fits[i]
                    best_overall = list(population[i])
            
            # If fully covered, try to shrink
            if best_fitness == len(best_overall):
                # All covered, just minimize size
                pass
            
            # Selection: tournament
            def select():
                a = random.randint(0, len(population) - 1)
                b = random.randint(0, len(population) - 1)
                return population[a] if fits[a] <= fits[b] else population[b]
            
            new_population = [list(best_overall)]  # Elitism
            
            while len(new_population) < pop_size:
                if time_left() < 5:
                    break
                
                parent1 = select()
                parent2 = select()
                
                # Crossover: random subset of parent1's codewords + fill from parent2
                # Strategy: take codewords that cover "hard" regions
                child = []
                child_set = set()
                
                # Start with a random mix
                for c in parent1:
                    if random.random() < 0.6:
                        child.append(c)
                        child_set.add(c)
                for c in parent2:
                    if c not in child_set and random.random() < 0.4:
                        child.append(c)
                        child_set.add(c)
                
                # Repair: ensure coverage
                if child:
                    counts_c = np.zeros(N, dtype=np.int32)
                    for c in child:
                        counts_c[neighbor_arrays[c]] += 1
                    uncovered = np.where(counts_c == 0)[0]
                    
                    if len(uncovered) > 0:
                        # Greedily add to cover gaps
                        for _ in range(min(len(uncovered), 30)):
                            if time_left() < 5:
                                break
                            unc = np.where(counts_c == 0)[0]
                            if len(unc) == 0:
                                break
                            best_c = -1
                            best_cov = 0
                            # Sample candidates for speed
                            candidates = random.sample(range(N), min(200, N))
                            for c in candidates:
                                if c in child_set:
                                    continue
                                cov = int(np.sum(counts_c[neighbor_arrays[c]] == 0))
                                if cov > best_cov:
                                    best_cov = cov
                                    best_c = c
                            if best_c != -1:
                                child.append(best_c)
                                child_set.add(best_c)
                                counts_c[neighbor_arrays[best_c]] += 1
                            else:
                                break
                    
                    # Remove redundant codewords
                    for i in range(len(child) - 1, -1, -1):
                        if time_left() < 5:
                            break
                        c = child[i]
                        if np.any(counts_c[neighbor_arrays[c]] == 1):
                            continue
                        child.pop(i)
                        child_set.discard(c)
                        counts_c[neighbor_arrays[c]] -= 1
                        # Check if removal created new uncovered
                        if np.any(counts_c == 0):
                            # Undo removal
                            child.insert(i, c)
                            child_set.add(c)
                            counts_c[neighbor_arrays[c]] += 1
                            break
                
                # Mutation: small random changes
                if child:
                    if random.random() < 0.5 and len(child) > 5:
                        # Remove a random codeword
                        idx = random.randint(0, len(child) - 1)
                        c_rem = child.pop(idx)
                        child_set.discard(c_rem)
                    if random.random() < 0.5:
                        # Add a random codeword
                        c_add = random.randint(0, N - 1)
                        if c_add not in child_set:
                            child.append(c_add)
                            child_set.add(c_add)
                
                new_population.append(child)
            
            population = new_population
        
        # Final repair and optimization on best
        best = list(best_overall)
        counts_best = np.zeros(N, dtype=np.int32)
        for c in best:
            counts_best[neighbor_arrays[c]] += 1
        
        if np.any(counts_best == 0):
            best = greedy_complete(best)
        
        best = local_search_remove(best)
        best = two_opt_local_search(best)
        best = local_search_remove(best)
        
        return best
    
    def simulated_annealing(code, max_iters=300):
        """Simulated annealing for further refinement."""
        code = list(code)
        code_set = set(code)
        counts = build_cover_counts(code)
        
        uncovered_count = int(np.sum(counts == 0))
        penalty = max(3 * len(code), 20)
        
        current_obj = len(code) + penalty * uncovered_count
        best_code = list(code)
        best_obj = current_obj
        
        temp = 15.0
        cooling = 0.985
        
        for it in range(max_iters):
            if time_left() < 4:
                break
            if temp < 0.1:
                break
            
            move_type = random.random()
            
            if move_type < 0.3 and len(code) > 10:
                idx = random.randint(0, len(code) - 1)
                c_remove = code[idx]
                newly_uncovered = int(np.sum(counts[neighbor_arrays[c_remove]] == 1))
                delta_obj = -1 + penalty * newly_uncovered
                new_obj = current_obj + delta_obj
                
                if new_obj < current_obj or (delta_obj > 0 and random.random() < np.exp(-delta_obj / temp)):
                    code.pop(idx)
                    code_set.discard(c_remove)
                    counts[neighbor_arrays[c_remove]] -= 1
                    current_obj = new_obj
                    uncovered_count += newly_uncovered
                    
            elif move_type < 0.6:
                c_add = random.randint(0, N - 1)
                if c_add in code_set:
                    continue
                newly_covered = int(np.sum(counts[neighbor_arrays[c_add]] == 0))
                delta_obj = 1 - penalty * newly_covered
                new_obj = current_obj + delta_obj
                
                if new_obj < current_obj or (delta_obj > 0 and random.random() < np.exp(-delta_obj / temp)):
                    code.append(c_add)
                    code_set.add(c_add)
                    counts[neighbor_arrays[c_add]] += 1
                    current_obj = new_obj
                    uncovered_count -= newly_covered
                    
            else:
                if len(code) == 0:
                    continue
                idx = random.randint(0, len(code) - 1)
                c_remove = code[idx]
                c_add = random.randint(0, N - 1)
                if c_add in code_set:
                    continue
                rem_uncov = int(np.sum(counts[neighbor_arrays[c_remove]] == 1))
                add_cov = int(np.sum(counts[neighbor_arrays[c_add]] == 0))
                uncov_delta = rem_uncov - add_cov
                delta_obj = penalty * uncov_delta
                new_obj = current_obj + delta_obj
                
                if new_obj < current_obj or (delta_obj > 0 and random.random() < np.exp(-delta_obj / temp)):
                    code[idx] = c_add
                    code_set.discard(c_remove)
                    code_set.add(c_add)
                    counts[neighbor_arrays[c_remove]] -= 1
                    counts[neighbor_arrays[c_add]] += 1
                    current_obj = new_obj
                    uncovered_count += uncov_delta
            
            if current_obj < best_obj:
                best_obj = current_obj
                best_code = list(code)
            
            temp *= cooling
        
        # Ensure full coverage
        counts_best = build_cover_counts(best_code)
        if np.any(counts_best == 0):
            best_code = greedy_complete(best_code)
        
        return best_code
    
    def flow_min_cover(initial_code):
        """
        Use push-relabel on a bipartite network to find minimum-size codeword set.
        
        Network: source -> codewords (cap 1) -> words (cap 1) -> sink (cap 1)
        Find max flow; if all 6561 words are saturated, the used codewords
        form a valid cover. Then prune redundant ones.
        
        We use a warm start: pre-select codewords from initial_code to bias
        the flow, then let push-relabel find the minimum.
        """
        # Build adjacency: for each codeword, which words it covers
        # We'll use a sparse representation for the bipartite graph
        
        # Precompute: for each word, which codewords can cover it (reverse adjacency)
        # word_to_codewords[w] = list of codeword indices c where w in neighbor_arrays[c]
        word_to_codewords = [[] for _ in range(N)]
        for c in range(N):
            for w in neighbor_arrays[c]:
                word_to_codewords[int(w)].append(c)
        
        # For push-relabel, we need:
        # Nodes: 0 = source, 1..N = codewords, N+1..2N = words, 2N+1 = sink
        # Total nodes: 2N + 2 = 13124
        # Edges: source->codeword (N edges), codeword->word (N*17 edges), word->sink (N edges)
        
        total_nodes = 2 * N + 2
        src = 0
        sink = total_nodes - 1
        
        # Build adjacency list with edge indices
        # Each edge: (to, cap, rev_idx)
        adj = [[] for _ in range(total_nodes)]
        
        def add_edge(u, v, cap):
            adj[u].append([v, cap, len(adj[v])])
            adj[v].append([u, 0, len(adj[u]) - 1])
        
        # Source -> codewords
        for c in range(N):
            add_edge(src, 1 + c, 1)
        
        # Codewords -> words
        for c in range(N):
            for w in neighbor_arrays[c]:
                w_int = int(w)
                add_edge(1 + c, N + 1 + w_int, 1)
        
        # Words -> sink
        for w in range(N):
            add_edge(N + 1 + w, sink, 1)
        
        # Push-relabel with gap heuristic and global relabel
        height = [0] * total_nodes
        excess = [0] * total_nodes
        seen = [0] * total_nodes  # current arc pointer
        count_height = [0] * (2 * total_nodes + 1)  # count of nodes at each height
        
        # Initialize
        height[src] = total_nodes
        count_height[0] = total_nodes - 1
        count_height[total_nodes] = 1
        
        # Pre-select: set initial flow on edges from source to initial codewords
        # This biases the algorithm
        initial_set = set(initial_code)
        
        # Actually, for minimum cover, we just need max flow = N
        # The number of codewords used = flow through source edges
        # Let's just run max flow and see how many codewords are used
        
        # Standard push-relabel
        # First, saturate source edges
        for edge in adj[src]:
            v, cap, rev = edge
            if cap > 0:
                flow = cap
                edge[1] -= flow
                adj[v][rev][1] += flow
                excess[v] += flow
                excess[src] -= flow
        
        active = deque()
        for v in range(1, total_nodes - 1):
            if excess[v] > 0:
                active.append(v)
        
        def global_relabel():
            """BFS from sink to set heights."""
            for i in range(total_nodes):
                height[i] = total_nodes
                seen[i] = 0
            height[sink] = 0
            bfs_q = deque([sink])
            # Reverse BFS: for each edge u->v with residual cap > 0, 
            # we can go from v to u in reverse
            # Actually, we need: height[u] <= height[v] + 1 for residual edges u->v
            # So from sink, follow reverse residual edges
            while bfs_q:
                v = bfs_q.popleft()
                # For each edge in adj[v], the reverse edge is in adj[to]
                # We want edges u->v where residual cap > 0
                # That means in adj[u], there's an edge to v with cap > 0
                # Equivalently, in adj[v], the reverse edge from v to u has cap > 0 (i.e., original flow > 0)
                # Hmm, let me think differently.
                # Residual graph: edge u->v exists if adj[u] has entry [v, cap>0, ...]
                # For global relabel, we want shortest path from each node to sink in residual graph.
                # So we BFS backwards from sink in the residual graph.
                # An edge u->v in residual graph means we can go from u to v.
                # For reverse BFS from sink: if there's a residual edge u->v and height[v] is known,
                # then height[u] can be at most height[v] + 1.
                # So we need to find all u such that there's a residual edge u->v.
                # That's: for each edge in adj[u] that points to v with cap > 0.
                # Equivalently: for each reverse edge in adj[v] (i.e., edge v->u in adj[v]),
                # the corresponding forward edge u->v has cap = adj[v][idx][1] (the reverse cap).
                
                # Let me just iterate all edges - for a graph this size it should be OK
                pass
            
            # Simpler: just do forward BFS from sink using reverse residual edges
            # For node v, its predecessors in residual graph are nodes u where adj[u] has edge to v with cap > 0
            # Which means adj[v] has a reverse edge back to u with cap = flow on u->v
            # So: for each edge [u, rev_cap, rev] in adj[v], if rev_cap > 0, then u has residual edge to v
            while bfs_q:
                v = bfs_q.popleft()
                for edge in adj[v]:
                    u = edge[0]
                    rev_cap = edge[1]  # This is the capacity of the reverse edge v->u, which equals flow on u->v
                    if rev_cap > 0 and height[u] == total_nodes:
                        height[u] = height[v] + 1
                        bfs_q.append(u)
        
        # Actually let me redo this more carefully
        def push_relabel():
            for i in range(total_nodes):
                height[i] = 0
                seen[i] = 0
            height[src] = total_nodes
            count_height = [0] * (2 * total_nodes + 1)
            count_height[0] = total_nodes - 1
            count_height[total_nodes] = 1
            
            # Saturate source edges
            for edge in adj[src]:
                v, cap, rev = edge
                if cap > 0:
                    flow = cap
                    edge[1] -= flow
                    adj[v][rev][1] += flow
                    excess[v] += flow
                    excess[src] -= flow
            
            active = deque()
            for v in range(1, total_nodes - 1):
                if excess[v] > 0:
                    active.append(v)
            
            def global_relabel():
                for i in range(total_nodes):
                    height[i] = total_nodes
                    seen[i] = 0
                height[sink] = 0
                bfs_q = deque([sink])
                while bfs_q:
                    v = bfs_q.popleft()
                    for edge in adj[v]:
                        u = edge[0]
                        # edge is v->u with capacity edge[1]
                        # The reverse edge u->v has capacity = adj[u][edge[2]][1]
                        # We want residual edges u->v, i.e., adj[u] has edge to v with cap > 0
                        # The reverse of edge v->u is in adj[u] at index edge[2]
                        # Its capacity is the flow on u->v
                        rev_edge = adj[u][edge[2]]
                        if rev_edge[1] > 0 and height[u] == total_nodes:
                            height[u] = height[v] + 1
                            bfs_q.append(u)
            
            gap_count = [0] * (2 * total_nodes + 1)
            gap_count[0] = total_nodes - 1
            gap_count[total_nodes] = 1
            
            work = 0
            max_work = 50000000  # safety limit
            
            while active and work < max_work:
                v = active.popleft()
                work += 1
                
                if excess[v] == 0:
                    continue
                
                discharged = False
                while excess[v] > 0 and not discharged:
                    if seen[v] < len(adj[v]):
                        edge = adj[v][seen[v]]
                        w = edge[0]
                        cap = edge[1]
                        if cap > 0 and height[v] == height[w] + 1:
                            # Push
                            flow = min(excess[v], cap)
                            edge[1] -= flow
                            adj[w][edge[2]][1] += flow
                            excess[v] -= flow
                            excess[w] += flow
                            if w != src and w != sink and excess[w] == flow:
                                active.append(w)
                        else:
                            seen[v] += 1
                    else:
                        # Relabel
                        min_h = 2 * total_nodes
                        for edge in adj[v]:
                            if edge[1] > 0:
                                min_h = min(min_h, height[edge[0]])
                        
                        if min_h < 2 * total_nodes:
                            new_h = min_h + 1
                        else:
                            new_h = 2 * total_nodes - 1
                        
                        # Gap heuristic
                        if gap_count[height[v]] == 1:
                            # Gap at height[v]: all nodes above can't reach sink
                            for i in range(total_nodes):
                                if i != src and i != sink and height[i] > height[v] and height[i] < total_nodes:
                                    gap_count[height[i]] -= 1
                                    height[i] = total_nodes + 1
                                    gap_count[total_nodes + 1] += 1
                                    seen[i] = 0
                            gap_count[height[v]] -= 1
                            height[v] = total_nodes + 1
                            gap_count[total_nodes + 1] += 1
                            seen[v] = 0
                            break
                        else:
                            gap_count[height[v]] -= 1
                            height[v] = new_h
                            gap_count[new_h] += 1
                            seen[v] = 0
            
            return sum(1 for edge in adj[src] if edge[1] == 0)  # number of codewords used
        
        # Run push-relabel
        n_codewords_used = push_relabel()
        
        # Extract which codewords were used (source edge fully saturated)
        used_codewords = []
        for c in range(N):
            edge = adj[src][c]  # source -> codeword c
            if edge[1] == 0:  # fully used
                used_codewords.append(c)
        
        # Verify coverage
        counts = np.zeros(N, dtype=np.int32)
        for c in used_codewords:
            counts[neighbor_arrays[c]] += 1
        
        if np.any(counts == 0):
            # Flow didn't saturate all words; fall back
            return None
        
        # Prune: remove redundant codewords
        code = local_search_remove(used_codewords)
        return code
    
    # Main algorithm
    # 1. Deterministic algebraic initialization
    code = algebraic_init()
    code = greedy_complete(code)
    
    # 2. Strict local search: removals, then 2-opt
    code = local_search_remove(code)
    code = two_opt_local_search(code)
    code = local_search_remove(code)
    
    best_code = code
    best_size = len(code)
    
    # 3. Flow-based optimization: find minimum cover via max flow
    if time_left() > 30:
        flow_code = flow_min_cover(best_code)
        if flow_code is not None and len(flow_code) < best_size:
            best_code = flow_code
            best_size = len(flow_code)
    
    # 4. Additional local search passes if time permits
    if time_left() > 20:
        code = two_opt_local_search(best_code)
        code = local_search_remove(code)
        if len(code) < best_size:
            best_code = code
            best_size = len(code)
    
    # 5. Second flow attempt with different initial code
    if time_left() > 25:
        # Perturb best code slightly and try flow again
        perturbed = list(best_code)
        random.shuffle(perturbed)
        # Remove a few random ones to create a different starting point
        for _ in range(min(5, len(perturbed) // 10)):
            if perturbed:
                perturbed.pop(random.randint(0, len(perturbed) - 1))
        flow_code2 = flow_min_cover(perturbed)
        if flow_code2 is not None and len(flow_code2) < best_size:
            best_code = flow_code2
            best_size = len(flow_code2)
    
    # 6. Simulated annealing for further refinement
    if time_left() > 12:
        sa_code = simulated_annealing(best_code, max_iters=150)
        sa_code = local_search_remove(sa_code)
        sa_code = two_opt_local_search(sa_code)
        sa_code = local_search_remove(sa_code)
        if len(sa_code) < best_size:
            best_code = sa_code
            best_size = len(sa_code)
    
    # 7. Final flow attempt
    if time_left() > 15:
        flow_code3 = flow_min_cover(best_code)
        if flow_code3 is not None and len(flow_code3) < best_size:
            best_code = flow_code3
            best_size = len(flow_code3)
    
    # 8. Final pass
    if time_left() > 5:
        best_code = local_search_remove(best_code)
        best_code = two_opt_local_search(best_code)
        best_code = local_search_remove(best_code)
    
    # Verify full coverage
    final_counts = build_cover_counts(best_code)
    if np.any(final_counts == 0):
        best_code = greedy_complete(best_code)
        best_code = local_search_remove(best_code)
    
    # Final deduplication safety
    seen = set()
    deduped = []
    for c in best_code:
        if c not in seen:
            seen.add(c)
            deduped.append(c)
    best_code = deduped
    
    # Convert indices to words
    result = []
    for c in best_code:
        result.append(idx_to_word(c))
    
    return result
