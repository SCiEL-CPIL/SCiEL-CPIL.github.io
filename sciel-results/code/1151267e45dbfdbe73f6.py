import itertools
import random
import time

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    total_words = q ** n
    # Map word tuple -> index
    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx
    
    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w
    
    # Precompute neighbors for each word index (word itself + words at distance <= radius)
    # For radius 1: the word itself + n*(q-1) neighbors
    neighbors = []
    for idx in range(total_words):
        w = idx_to_word(idx)
        nbrs = [idx]
        for pos in range(n):
            orig = w[pos]
            for val in range(q):
                if val == orig:
                    continue
                w[pos] = val
                nbrs.append(word_to_idx(tuple(w)))
            w[pos] = orig
        neighbors.append(nbrs)
    
    def coverage_of_codeword_set(codewords):
        """Return coverage count array for a set of codeword indices."""
        cov = [0] * total_words
        for ci in codewords:
            for nb in neighbors[ci]:
                cov[nb] += 1
        return cov
    
    def is_covered(cov):
        """Check if all words are covered (cov[i] >= 1 for all i)."""
        for i in range(total_words):
            if cov[i] == 0:
                return False
        return True
    
    def greedy_remove_redundant(codewords):
        """Remove redundant codewords. Returns new list and count removed."""
        codewords = list(codewords)
        removed = 0
        changed = True
        while changed:
            changed = False
            cov = coverage_of_codeword_set(codewords)
            for i in range(len(codewords)):
                if not is_covered(cov):
                    break
                # Try removing codeword i
                ci = codewords[i]
                for nb in neighbors[ci]:
                    cov[nb] -= 1
                if is_covered(cov):
                    # Safe to remove
                    codewords.pop(i)
                    removed += 1
                    changed = True
                    break
                else:
                    # Restore
                    for nb in neighbors[ci]:
                        cov[nb] += 1
        return codewords, removed
    
    def greedy_greedy_add(codewords, max_time=None):
        """Add codewords greedily to cover all uncovered words."""
        codewords = list(codewords)
        cov = coverage_of_codeword_set(codewords)
        # Find uncovered words
        uncovered = [i for i in range(total_words) if cov[i] == 0]
        if not uncovered:
            return codewords
        # For each candidate codeword, count how many uncovered words it covers
        # Use a priority approach
        while uncovered:
            if max_time and time.time() > max_time:
                break
            best_ci = -1
            best_count = 0
            # Sample candidates if too many
            candidates = list(range(total_words))
            random.shuffle(candidates)
            for ci in candidates:
                count = 0
                for nb in neighbors[ci]:
                    if cov[nb] == 0:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_ci = ci
                if best_count == len(uncovered):
                    break
            if best_ci == -1:
                break
            codewords.append(best_ci)
            for nb in neighbors[best_ci]:
                cov[nb] += 1
            uncovered = [i for i in uncovered if cov[i] == 0]
        return codewords
    
    def k_swap_perturb(codewords, k):
        """Remove k random codewords and add k new random ones."""
        codewords = list(codewords)
        if len(codewords) < k:
            k = len(codewords)
        to_remove = random.sample(range(len(codewords)), k)
        to_remove_set = set(to_remove)
        new_codewords = [c for i, c in enumerate(codewords) if i not in to_remove_set]
        # Add k random new codewords
        for _ in range(k):
            new_codewords.append(random.randint(0, total_words - 1))
        return new_codewords
    
    # Multi-start ILS
    best_code = None
    best_len = float('inf')
    
    # Initial solution: greedy set cover
    init_codewords = []
    cov = [0] * total_words
    uncovered = set(range(total_words))
    while uncovered:
        if time.time() > deadline:
            break
        best_ci = -1
        best_count = 0
        candidates = list(range(total_words))
        random.shuffle(candidates)
        for ci in candidates:
            count = 0
            for nb in neighbors[ci]:
                if cov[nb] == 0:
                    count += 1
            if count > best_count:
                best_count = count
                best_ci = ci
            if best_count == len(uncovered):
                break
        if best_ci == -1:
            break
        init_codewords.append(best_ci)
        for nb in neighbors[best_ci]:
            cov[nb] += 1
            if cov[nb] == 1:
                uncovered.discard(nb)
    
    # Remove redundant
    init_codewords, _ = greedy_remove_redundant(init_codewords)
    
    if len(init_codewords) < best_len:
        best_len = len(init_codewords)
        best_code = list(init_codewords)
    
    # ILS loop
    current = list(init_codewords)
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        # Perturbation
        k = random.choice([1, 2, 3])
        perturbed = k_swap_perturb(current, k)
        
        # Local search: greedy remove redundant
        improved, _ = greedy_remove_redundant(perturbed)
        
        if len(improved) < len(current):
            current = improved
            if len(current) < best_len:
                best_len = len(current)
                best_code = list(current)
        else:
            # Occasionally accept non-improving moves (simulated annealing style)
            if random.random() < 0.1:
                current = improved
    
    # Convert indices back to words
    if best_code is None:
        # Fallback: use the direct sum construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    return [idx_to_word(ci) for ci in best_code]
