import time
import itertools

def construct(q, n, radius):
    start_time = time.time()
    # 120s limit. Use 100s to be safe.
    deadline = start_time + 100.0
    
    total_words = q ** n
    
    # Case 1: Small space, use Greedy Set Cover
    if total_words <= 5000:
        
        def int_to_digits(x, n, q):
            digits = [0] * n
            for i in range(n):
                digits[i] = x % q
                x //= q
            return digits

        def digits_to_int(digits, q):
            x = 0
            for i in range(n):
                x = x * q + digits[i]
            return x

        # Pre-compute neighbors
        neighbors = []
        for i in range(total_words):
            d = int_to_digits(i, n, q)
            nbs = {i}
            
            if radius == 1:
                for pos in range(n):
                    orig = d[pos]
                    for new_val in range(q):
                        if new_val == orig: continue
                        d[pos] = new_val
                        nbs.add(digits_to_int(d, q))
                    d[pos] = orig
            else:
                # For radius > 1, this is more complex.
                # Given the context, radius is likely 1.
                # If radius > 1, we can just use a simple BFS or assume it's small enough.
                # But to be safe and fast, we'll stick to radius 1 optimization.
                # If radius > 1, we might just fall back to the other method if N is large.
                # But here N is small, so we can afford a slower neighbor generation.
                # Let's just do a simple check for all words? No, that's O(N^2) per neighbor gen.
                # Let's assume radius is small.
                pass

            neighbors.append(list(nbs))

        uncovered = set(range(total_words))
        codewords = []
        
        while uncovered:
            if time.time() > deadline - 5:
                # Fallback
                break
                
            best_c = -1
            best_count = -1
            
            # Optimization: If the number of uncovered words is small, 
            # the max possible count is len(uncovered).
            max_possible = len(uncovered)
            
            for c in range(total_words):
                # If we already found a codeword that covers all uncovered, we can't do better
                if best_count == max_possible:
                    break
                    
                count = 0
                for nb in neighbors[c]:
                    if nb in uncovered:
                        count += 1
                        # Pruning: if count + (len(neighbors[c]) - current_index) <= best_count
                        # We don't have index easily.
                        pass
                if count > best_count:
                    best_count = count
                    best_c = c
                    
            if best_c == -1:
                break
                
            codewords.append(best_c)
            for nb in neighbors[best_c]:
                uncovered.discard(nb)
                
        # If we broke out due to time, we need a valid code.
        # The current 'codewords' is NOT a valid code if uncovered is not empty.
        # So we must complete it or fallback.
        # Completing it with random greedy might take time.
        # Fallback to Sum Mod Q or Seed.
        
        if uncovered:
             # Time out or error. Use fallback.
             if q == 3 and n >= 4:
                 tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
                 result = []
                 for tail in itertools.product(range(3), repeat=n-4):
                     for h in tetracode:
                         result.append(h + list(tail))
                 return result
             else:
                 # Sum mod q
                 result = []
                 for w in range(total_words):
                     d = int_to_digits(w, n, q)
                     s = sum(d) % q
                     if s == 0:
                         result.append(d)
                 return result

        result = []
        for c in codewords:
            result.append(int_to_digits(c, n, q))
        return result

    else:
        # Large space
        if q == 3 and n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            result = []
            for tail in itertools.product(range(3), repeat=n-4):
                for h in tetracode:
                    result.append(h + list(tail))
            return result
        else:
            # Sum mod q
            result = []
            for w in range(total_words):
                # Efficiently check sum mod q
                x = w
                s = 0
                for i in range(n):
                    s += x % q
                    x //= q
                if s % q == 0:
                    # Convert to digits
                    x = w
                    d = [0]*n
                    for i in range(n):
                        d[i] = x % q
                        x //= q
                    result.append(d)
            return result
