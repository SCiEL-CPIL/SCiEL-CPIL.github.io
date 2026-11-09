import time
import random

def construct():
        start_time = time.time()
        deadline = start_time + 55
        
        candidates = list(range(2, 129))
        
        def is_prime(n):
            if n < 2: return False
            for i in range(2, int(n**0.5)+1):
                if n % i == 0:
                    return False
            return True
            
        primes = [x for x in candidates if is_prime(x)]
        composites = [x for x in candidates if not is_prime(x)]
        
        def get_products(seq):
            prods = set()
            for i in range(len(seq)):
                p = 1
                for j in range(i, len(seq)):
                    p *= seq[j]
                    prods.add(p)
            return prods

        def try_insertion(seq, c):
            # Returns (new_seq, success)
            old_prods = get_products(seq)
            if c in old_prods:
                return None, False
                
            for i in range(len(seq) + 1):
                if i > 0 and seq[i-1] >= c:
                    continue
                if i < len(seq) and c >= seq[i]:
                    continue
                
                # Calculate new products
                new_prods = set()
                valid = True
                # Left suffixes
                left_sufs = [1]
                p = 1
                for k in range(i-1, -1, -1):
                    p *= seq[k]
                    left_sufs.append(p)
                    
                # Right prefixes
                right_pres = [1]
                p = 1
                for k in range(i, len(seq)):
                    p *= seq[k]
                    right_pres.append(p)
                    
                for ls in left_sufs:
                    for rp in right_pres:
                        prod = ls * c * rp
                        if prod in old_prods:
                            valid = False
                            break
                        if prod in new_prods:
                            valid = False
                            break
                        new_prods.add(prod)
                    if not valid:
                        break
                
                if valid:
                    return seq[:i] + [c] + seq[i:], True
            return None, False

        best_seq = primes
        best_len = len(best_seq)
        
        # Multiple restarts
        for _ in range(50):
            if time.time() > deadline - 5:
                break
                
            current_seq = primes[:]
            remaining = composites[:]
            random.shuffle(remaining)
            
            progress = True
            while progress:
                progress = False
                for c in remaining:
                    if time.time() > deadline - 2:
                        break
                    new_seq, success = try_insertion(current_seq, c)
                    if success:
                        current_seq = new_seq
                        remaining.remove(c)
                        if len(current_seq) > best_len:
                            best_seq = current_seq[:]
                            best_len = len(current_seq)
                        progress = True
                        break # Restart loop with new sequence
                        
        return best_seq
