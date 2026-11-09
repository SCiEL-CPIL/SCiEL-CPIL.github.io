import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit
    
    # Precompute prime factorization vectors for numbers 2..128
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(math.isqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    primes = [x for x in range(2, 129) if is_prime(x)]
    num_primes = len(primes)
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    # Exponent vector for each number 2..128
    exp_vecs = {}
    for n in range(2, 129):
        vec = [0] * num_primes
        temp = n
        for pi, p in enumerate(primes):
            if p * p > temp:
                break
            while temp % p == 0:
                vec[pi] += 1
                temp //= p
        if temp > 1:
            vec[prime_to_idx[temp]] += 1
        exp_vecs[n] = tuple(vec)
        
    # Packed integer representation for fast hashing
    # Max exponent for any prime in a block product:
    # Max number is 128 = 2^7.
    # Max length of sequence is 127.
    # Max exponent for 2 is 7 * 127 = 889.
    # 10 bits is enough (2^10 = 1024).
    BITS = 10
    MASK = (1 << BITS) - 1
    
    def pack_vec(vec):
        res = 0
        for i, e in enumerate(vec):
            res |= e << (i * BITS)
        return res

    def check_valid(seq):
        """Check if sequence has all distinct consecutive block products using prefix sums."""
        n = len(seq)
        if n == 0:
            return True
        # Build prefix sum vectors: prefix[i] = sum of exp_vecs[seq[0:i]]
        # prefix[0] is zero vector
        prefix_packed = [0] * (n + 1)
        current_vec = [0] * num_primes
        
        for i in range(n):
            ej = exp_vecs[seq[i]]
            for k in range(num_primes):
                current_vec[k] += ej[k]
            prefix_packed[i+1] = pack_vec(current_vec)
        
        prods = set()
        for i in range(n):
            pi_packed = prefix_packed[i]
            # We need to unpack pi to subtract? No, we can just compute the difference vector on the fly if we stored vectors, 
            # but we stored packed. 
            # Actually, subtraction in packed form is tricky if there are borrows. 
            # Since all differences are non-negative (prefix is increasing), we can unpack, subtract, and repack.
            # Or better: store the actual vectors for prefix? That's O(N * P) memory, which is small.
            
            # Let's optimize: store prefix vectors as lists of ints.
            pass
            
        # Revised check_valid for speed: store prefix vectors as lists
        prefix_vecs = [[0] * num_primes]
        curr = [0] * num_primes
        for val in seq:
            v = exp_vecs[val]
            for k in range(num_primes):
                curr[k] += v[k]
            prefix_vecs.append(curr[:]) # Copy
            
        prods = set()
        for i in range(n):
            pi = prefix_vecs[i]
            for j in range(i + 1, n + 1):
                pj = prefix_vecs[j]
                diff = [pj[k] - pi[k] for k in range(num_primes)]
                packed = pack_vec(diff)
                if packed in prods:
                    return False
                prods.add(packed)
        return True

    def try_insert(seq, c, pos):
        """Try inserting c at position pos in seq. Return new seq if valid and ordered, else None."""
        if pos < 0 or pos > len(seq):
            return None
        if pos > 0 and seq[pos-1] >= c:
            return None
        if pos < len(seq) and c >= seq[pos]:
            return None
            
        new_seq = seq[:pos] + [c] + seq[pos:]
        if check_valid(new_seq):
            return new_seq
        return None

    def local_search(seq, max_iters=1000):
        """Improve sequence by insertions, swaps, and moves."""
        current = seq[:]
        used = set(current)
        remaining = [x for x in range(2, 129) if x not in used]
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            improved = False
            
            # 1. Try Insertions
            for c in remaining:
                # Try inserting at each valid position
                for i in range(len(current) + 1):
                    new_seq = try_insert(current, c, i)
                    if new_seq is not None:
                        current = new_seq
                        used = set(current)
                        remaining = [x for x in range(2, 129) if x not in used]
                        improved = True
                        break
                if improved:
                    break
            
            # 2. Try Swaps (Remove one, add another)
            if not improved:
                for i in range(len(current)):
                    removed = current[i]
                    base = current[:i] + current[i+1:]
                    # We can add any number from remaining, or even one that was removed? 
                    # The task says "strictly increasing integers in [2,128]". It doesn't say we must use a specific set.
                    # But maximizing length suggests using as many distinct numbers as possible.
                    # So swapping usually means replacing a "bad" number with a "good" one from remaining.
                    for c in remaining:
                        for j in range(len(base) + 1):
                            new_seq = try_insert(base, c, j)
                            if new_seq is not None:
                                # Ensure length doesn't decrease? 
                                # len(new_seq) == len(base) + 1 == len(current)
                                # So length is maintained.
                                current = new_seq
                                used = set(current)
                                remaining = [x for x in range(2, 129) if x not in used]
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
            
            # 3. Try Moves (Move an existing element to a different position)
            if not improved:
                for i in range(len(current)):
                    val = current[i]
                    base = current[:i] + current[i+1:]
                    for j in range(len(base) + 1):
                        if i < j:
                            j_adj = j + 1 # Adjusting for removal? 
                            # Actually, try_insert handles the logic. 
                            # But we need to ensure we don't just put it back where it was.
                            # If j == i or j == i+1, it's effectively no move or a swap with neighbor?
                            # try_insert checks order.
                            new_seq = try_insert(base, val, j)
                            if new_seq is not None and new_seq != current:
                                current = new_seq
                                improved = True
                                break
                    if improved:
                        break
            
            if not improved:
                break
                
        return current

    # 1. Initial Greedy Construction
    # We try a few different greedy orders to get a good starting point
    best_seq = []
    
    # Standard ascending greedy
    seq1 = []
    for n in range(2, 129):
        if time.time() > deadline:
            break
        # Try appending to end first (fastest check)
        if not seq1 or seq1[-1] < n:
             new_seq = try_insert(seq1, n, len(seq1))
             if new_seq is not None:
                 seq1 = new_seq
    best_seq = seq1
    
    # 2. Iterated Local Search
    rng = random.Random(42)
    
    while time.time() < deadline - 5: # Leave 5s for final polish
        # Local Search
        current_best = local_search(best_seq, max_iters=500)
        
        if len(current_best) > len(best_seq):
            best_seq = current_best
        else:
            # If no improvement, keep best_seq as base for perturbation
            pass
            
        # Perturbation: Kick
        # Remove 2-5 random elements
        kick_size = rng.randint(2, 5)
        if len(best_seq) > kick_size:
            kick_indices = rng.sample(range(len(best_seq)), kick_size)
            kick_indices.sort(reverse=True)
            
            perturbed = best_seq[:]
            for idx in kick_indices:
                perturbed.pop(idx)
                
            # Repair: Greedily add back best possible numbers
            # We want to restore length if possible
            used = set(perturbed)
            remaining = [x for x in range(2, 129) if x not in used]
            
            # Sort remaining by some heuristic? 
            # Maybe prefer numbers that were in the original best_seq?
            original_set = set(best_seq)
            remaining.sort(key=lambda x: (x not in original_set, x))
            
            for c in remaining:
                if time.time() > deadline - 5:
                    break
                # Try inserting at any position
                for i in range(len(perturbed) + 1):
                    new_seq = try_insert(perturbed, c, i)
                    if new_seq is not None:
                        perturbed = new_seq
                        used = set(perturbed)
                        break
                        
            # If perturbed is longer than best_seq, update
            if len(perturbed) > len(best_seq):
                best_seq = perturbed
            else:
                # Even if not longer, maybe it's a different local optimum?
                # We can try local search on it
                # But to save time, only if it's close in length
                if len(perturbed) >= len(best_seq) - 2:
                     improved_pert = local_search(perturbed, max_iters=200)
                     if len(improved_pert) > len(best_seq):
                         best_seq = improved_pert

    # Final Polish: Try to squeeze out more length with more iterations
    final_seq = local_search(best_seq, max_iters=2000)
    if len(final_seq) > len(best_seq):
        best_seq = final_seq
        
    return best_seq
