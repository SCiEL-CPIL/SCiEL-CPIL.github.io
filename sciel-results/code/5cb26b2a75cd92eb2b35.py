import itertools
import time
import random
import sys

def _standard_superperm(n):
    """Generate standard superpermutation for n using recursive construction."""
    if n == 1:
        return [1]
    if n == 2:
        return [1, 2, 1]
    if n == 3:
        return [1, 2, 3, 1, 2, 1, 3, 2, 1]
    
    prev = _standard_superperm(n - 1)
    
    seen = set()
    order = []
    for i in range(len(prev) - (n - 1) + 1):
        window = tuple(prev[i:i + n - 1])
        if window not in seen:
            seen.add(window)
            if sorted(window) == list(range(1, n)):
                order.append(window)
    
    new = []
    for perm in order:
        block = list(perm) + [n] + list(perm)
        max_ov = min(len(new), len(block))
        overlap = 0
        for ov in range(max_ov, -1, -1):
            if ov == 0 or new[len(new) - ov:] == block[:ov]:
                overlap = ov
                break
        new.extend(block[overlap:])
    
    return new

def _verify_superpermutation(word, n):
    """Check if word contains all permutations of 1..n as consecutive subsequences."""
    all_perms = set(itertools.permutations(range(1, n + 1)))
    found = set()
    for i in range(len(word) - n + 1):
        p = tuple(word[i:i + n])
        if p in all_perms:
            found.add(p)
        if len(found) == len(all_perms):
            return True
    return len(found) == len(all_perms)

def _construct_from_order(order, n):
    """
    Construct superpermutation for n from an order of (n-1)-permutations,
    inserting symbol n with maximum overlap.
    """
    new = []
    for perm in order:
        block = list(perm) + [n] + list(perm)
        max_ov = min(len(new), len(block))
        overlap = 0
        for ov in range(max_ov, -1, -1):
            if ov == 0 or new[len(new) - ov:] == block[:ov]:
                overlap = ov
                break
        new.extend(block[overlap:])
    return new

def _extract_order_from_word(word, n):
    """Extract the order of (n-1)-permutations from a superpermutation for n."""
    seen = set()
    order = []
    for i in range(len(word) - (n - 1) + 1):
        window = tuple(word[i:i + n - 1])
        if window not in seen:
            seen.add(window)
            if sorted(window) == list(range(1, n)):
                order.append(window)
    return order

def _build_overlap_matrix(perms, n):
    """
    Precompute overlap matrix where overlap[i][j] is the maximum overlap
    between the end of block_i and the start of block_j.
    block_i = perms[i] + [n] + perms[i]
    """
    N = len(perms)
    blocks = [list(p) + [n] + list(p) for p in perms]
    overlap_mat = [[0] * N for _ in range(N)]
    
    for i in range(N):
        bi = blocks[i]
        for j in range(N):
            if i == j:
                overlap_mat[i][j] = len(bi)
                continue
            bj = blocks[j]
            max_ov = min(len(bi), len(bj))
            ov = 0
            for k in range(max_ov, -1, -1):
                if k == 0 or bi[len(bi) - k:] == bj[:k]:
                    ov = k
                    break
            overlap_mat[i][j] = ov
    
    return overlap_mat

def _path_length(path, overlap_mat, block_len):
    """Compute total length of word built from path given overlap matrix."""
    total = block_len
    for i in range(len(path) - 1):
        total += block_len - overlap_mat[path[i]][path[i + 1]]
    return total

def _genetic_algorithm(perms, n, deadline):
    """
    Genetic algorithm to find a good ordering of permutations.
    """
    N = len(perms)
    block_len = 2 * (n - 1) + 1  # len(perm) + 1 + len(perm)
    
    # Build overlap matrix
    overlap_mat = _build_overlap_matrix(perms, n)
    
    # Initial population
    pop_size = 50
    population = []
    
    # Standard order
    std_order = list(range(N))
    population.append(std_order)
    
    # Nearest neighbor from standard start
    nn_path = [0]
    unvisited = set(range(1, N))
    while unvisited:
        last = nn_path[-1]
        best_idx = min(unvisited, key=lambda x: -overlap_mat[last][x])
        nn_path.append(best_idx)
        unvisited.remove(best_idx)
    population.append(nn_path)
    
    # Random paths
    random.seed(42)
    for _ in range(pop_size - 2):
        p = list(range(N))
        random.shuffle(p)
        population.append(p)
    
    # Evaluate initial population
    def evaluate(path):
        return _path_length(path, overlap_mat, block_len)
    
    scored_pop = [(evaluate(p), p) for p in population]
    scored_pop.sort(key=lambda x: x[0])
    
    best_score, best_path = scored_pop[0]
    
    # GA loop
    mutation_rate = 0.1
    crossover_rate = 0.8
    elite_size = 5
    
    while time.time() < deadline:
        new_population = []
        
        # Elitism
        for i in range(min(elite_size, len(scored_pop))):
            new_population.append(scored_pop[i][1])
        
        # Generate offspring
        while len(new_population) < pop_size:
            # Tournament selection
            def select():
                idx1 = random.randint(0, len(scored_pop) - 1)
                idx2 = random.randint(0, len(scored_pop) - 1)
                if scored_pop[idx1][0] < scored_pop[idx2][0]:
                    return scored_pop[idx1][1]
                else:
                    return scored_pop[idx2][1]
            
            parent1 = select()
            parent2 = select()
            
            if random.random() < crossover_rate:
                # Order-based crossover
                child = [None] * N
                # Pick random positions from parent1
                pos1 = random.sample(range(N), N // 2)
                for pos in pos1:
                    child[pos] = parent1[pos]
                
                # Fill remaining from parent2 in order
                filled = set(child)
                idx2 = 0
                for i in range(N):
                    if child[i] is None:
                        while parent2[idx2] in filled:
                            idx2 += 1
                        child[i] = parent2[idx2]
                        filled.add(child[i])
            else:
                child = parent1[:]
            
            # Mutation
            if random.random() < mutation_rate:
                # 2-opt mutation
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                child[i+1:j+1] = child[i+1:j+1][::-1]
            
            # Swap mutation
            if random.random() < 0.3:
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                child[i], child[j] = child[j], child[i]
            
            new_population.append(child)
        
        # Evaluate new population
        scored_pop = [(evaluate(p), p) for p in new_population]
        scored_pop.sort(key=lambda x: x[0])
        
        if scored_pop[0][0] < best_score:
            best_score = scored_pop[0][0]
            best_path = scored_pop[0][1]
    
    return best_path, best_score

def _local_search(path, overlap_mat, block_len, deadline, max_iterations=10000):
    """
    Local search refinement using 2-opt and swap moves.
    """
    N = len(path)
    current_score = _path_length(path, overlap_mat, block_len)
    best_score = current_score
    best_path = path[:]
    
    iterations = 0
    while time.time() < deadline and iterations < max_iterations:
        iterations += 1
        improved = False
        
        # Try 2-opt moves
        for i in range(N - 2):
            if time.time() > deadline:
                return best_path, best_score
            for j in range(i + 1, N - 1):
                # Compute delta for 2-opt
                old_ov1 = overlap_mat[path[i]][path[i + 1]]
                old_ov2 = overlap_mat[path[j]][path[j + 1]]
                new_ov1 = overlap_mat[path[i]][path[j]]
                new_ov2 = overlap_mat[path[i + 1]][path[j + 1]]
                
                if new_ov1 + new_ov2 > old_ov1 + old_ov2:
                    path[i + 1:j + 1] = path[i + 1:j + 1][::-1]
                    current_score += (old_ov1 + old_ov2) - (new_ov1 + new_ov2)
                    if current_score < best_score:
                        best_score = current_score
                        best_path = path[:]
                    improved = True
                    break
            if improved:
                break
        
        # Try swap moves if no 2-opt improvement
        if not improved:
            for i in range(N - 1):
                if time.time() > deadline:
                    return best_path, best_score
                for j in range(i + 1, N):
                    # Compute delta for swap
                    if i == 0 and j == N - 1:
                        old_ov = overlap_mat[path[i]][path[j]]
                        new_ov = overlap_mat[path[j]][path[i]]
                        if new_ov > old_ov:
                            path[i], path[j] = path[j], path[i]
                            current_score += old_ov - new_ov
                            if current_score < best_score:
                                best_score = current_score
                                best_path = path[:]
                            improved = True
                            break
                    elif i == 0:
                        old_ov1 = overlap_mat[path[i]][path[i + 1]]
                        old_ov2 = overlap_mat[path[j]][path[j + 1]]
                        new_ov1 = overlap_mat[path[j]][path[i + 1]]
                        new_ov2 = overlap_mat[path[i]][path[j + 1]]
                        if new_ov1 + new_ov2 > old_ov1 + old_ov2:
                            path[i], path[j] = path[j], path[i]
                            current_score += (old_ov1 + old_ov2) - (new_ov1 + new_ov2)
                            if current_score < best_score:
                                best_score = current_score
                                best_path = path[:]
                            improved = True
                            break
                    elif j == N - 1:
                        old_ov1 = overlap_mat[path[i]][path[i + 1]]
                        old_ov2 = overlap_mat[path[N - 2]][path[N - 1]]
                        new_ov1 = overlap_mat[path[j]][path[i + 1]]
                        new_ov2 = overlap_mat[path[N - 2]][path[i]]
                        if new_ov1 + new_ov2 > old_ov1 + old_ov2:
                            path[i], path[j] = path[j], path[i]
                            current_score += (old_ov1 + old_ov2) - (new_ov1 + new_ov2)
                            if current_score < best_score:
                                best_score = current_score
                                best_path = path[:]
                            improved = True
                            break
                    else:
                        old_ov1 = overlap_mat[path[i - 1]][path[i]]
                        old_ov2 = overlap_mat[path[i]][path[i + 1]]
                        old_ov3 = overlap_mat[path[j]][path[j + 1]]
                        old_ov4 = overlap_mat[path[j + 1]][path[j + 2]] if j + 2 < N else 0
                        # This is getting complex, skip for now
                        pass
                if improved:
                    break
        
        if not improved:
            break
    
    return best_path, best_score

def construct(n):
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin before 150s limit
    
    # For n <= 5, return standard (which is optimal for n<=3, near-optimal for n=4,5)
    if n <= 5:
        word = _standard_superperm(n)
        return "".join(map(str, word))
    
    # For n=6,7: build from standard base, then optimize the ordering step
    # Step 1: Generate standard superpermutation for n-1
    base = _standard_superperm(n - 1)
    
    # Step 2: Extract the order of (n-1)-permutations
    order = _extract_order_from_word(base, n - 1)
    
    # Step 3: Optimize the order using genetic algorithm + local search
    if time.time() < deadline - 10:
        ga_deadline = deadline - 5
        path, ga_score = _genetic_algorithm(order, n, ga_deadline)
        
        # Build overlap matrix for local search
        block_len = 2 * (n - 1) + 1
        overlap_mat = _build_overlap_matrix(order, n)
        
        # Local search refinement
        if time.time() < deadline - 2:
            ls_deadline = deadline - 1
            path, ls_score = _local_search(path, overlap_mat, block_len, ls_deadline)
        
        # Convert path back to permutation order
        optimized_order = [order[i] for i in path]
    else:
        optimized_order = order
    
    # Step 4: Construct the final superpermutation
    word = _construct_from_order(optimized_order, n)
    
    # Verify
    if not _verify_superpermutation(word, n):
        # Fallback to standard
        word = _standard_superperm(n)
    
    return "".join(map(str, word))

# Quick self-test
if __name__ == "__main__":
    for n in range(1, 6):
        w = construct(n)
        valid = _verify_superpermutation(list(map(int, w)), n)
        print(f"n={n}, len={len(w)}, valid={valid}")
    
    # For n=6 and 7, just check timing and validity
    for n in [6, 7]:
        start = time.time()
        w = construct(n)
        elapsed = time.time() - start
        valid = _verify_superpermutation(list(map(int, w)), n)
        print(f"n={n}, len={len(w)}, valid={valid}, time={elapsed:.2f}s")
