import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140
    
    total = q ** n  # 729
    
    def int_to_word(val):
        w = []
        for _ in range(n):
            w.append(val % q)
            val //= q
        return w[::-1]
    
    # Precompute coverage: covers[i] = list of words covered by word i
    covers = []
    for i in range(total):
        w = int_to_word(i)
        covered = [i]
        for pos in range(n):
            for sym in range(q):
                if sym != w[pos]:
                    new_w = w[:]
                    new_w[pos] = sym
                    val = 0
                    for s in new_w:
                        val = val * q + s
                    covered.append(val)
        covers.append(covered)
    
    # Greedy set cover
    uncovered = [True] * total
    uncovered_count = total
    codewords = []
    
    while uncovered_count > 0:
        if time.time() > deadline - 5:
            break
        best_word = -1
        best_gain = -1
        for i in range(total):
            gain = 0
            for j in covers[i]:
                if uncovered[j]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_word = i
        if best_word < 0 or best_gain == 0:
            break
        codewords.append(best_word)
        for j in covers[best_word]:
            if uncovered[j]:
                uncovered[j] = False
                uncovered_count -= 1
    
    # Local search with coverage counts
    def make_counts(cw):
        counts = [0] * total
        for c in cw:
            for j in covers[c]:
                counts[j] += 1
        return counts
    
    def is_valid_counts(counts):
        for c in counts:
            if c == 0:
                return False
        return True
    
    def try_reduce(cw):
        """Remove redundant codewords using coverage counts."""
        result = list(cw)
        counts = make_counts(result)
        improved = True
        while improved:
            improved = False
            for i in range(len(result)):
                # Try removing result[i]
                c = result[i]
                for j in covers[c]:
                    counts[j] -= 1
                # Check if all still covered
                valid = True
                for j in covers[c]:
                    if counts[j] == 0:
                        valid = False
                        break
                if valid:
                    result.pop(i)
                    improved = True
                    break
                else:
                    # Restore
                    for j in covers[c]:
                        counts[j] += 1
        return result
    
    codewords = try_reduce(codewords)
    best = list(codewords)
    best_len = len(best)
    
    random.seed(42)
    temperature = 1.0
    cooling = 0.99
    
    current = list(codewords)
    
    while time.time() < deadline:
        # Generate neighbor
        r = random.random()
        if r < 0.5 and len(current) > 1:
            # Swap
            neighbor = list(current)
            idx = random.randint(0, len(neighbor) - 1)
            new_word = random.randint(0, total - 1)
            if new_word not in neighbor:
                neighbor[idx] = new_word
        elif r < 0.75:
            # Add
            neighbor = list(current)
            new_word = random.randint(0, total - 1)
            if new_word not in neighbor:
                neighbor.append(new_word)
        else:
            # Remove
            if len(current) <= 1:
                continue
            neighbor = list(current)
            idx = random.randint(0, len(neighbor) - 1)
            neighbor.pop(idx)
        
        # Check validity
        counts = make_counts(neighbor)
        if not is_valid_counts(counts):
            continue
        
        # Try to reduce
        neighbor = try_reduce(neighbor)
        neighbor_len = len(neighbor)
        current_len = len(current)
        
        delta = neighbor_len - current_len
        if delta < 0 or (delta == 0 and random.random() < 0.3) or (delta > 0 and random.random() < 2.718 ** (-delta / temperature)):
            current = neighbor
            if neighbor_len < best_len:
                best = list(current)
                best_len = neighbor_len
                temperature = 1.0
        
        temperature *= cooling
        if temperature < 0.01:
            temperature = 1.0
    
    return [int_to_word(w) for w in best]
