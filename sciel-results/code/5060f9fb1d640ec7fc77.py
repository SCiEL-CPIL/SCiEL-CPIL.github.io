import time
import random

def construct():
    start_time = time.time()
    deadline = 50.0  # Safety margin for 60s limit
    
    # Known valid set of size 31
    fallback = [226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256]
    
    def check_valid(s):
        """Check if set has cardinality separation property."""
        s = sorted(s)
        n = len(s)
        for k in range(1, n):
            max_sum_k = sum(s[n-k:])
            min_sum_k1 = sum(s[:k+1])
            if min_sum_k1 <= max_sum_k:
                return False
        return True
    
    def min_gap(s):
        s = sorted(s)
        n = len(s)
        min_g = float('inf')
        for k in range(1, n):
            max_sum_k = sum(s[n-k:])
            min_sum_k1 = sum(s[:k+1])
            gap = min_sum_k1 - max_sum_k
            if gap < min_g:
                min_g = gap
        return min_g
    
    # Try to find a valid set of size 32 or larger
    # Strategy: use a geometric-like progression with carefully chosen gaps
    
    # Approach 1: Try different base APs and perturb
    best_set = None
    best_gap = -float('inf')
    
    random.seed(42)
    
    # Try multiple initial configurations
    for trial in range(200):
        if time.time() > deadline:
            break
        
        # Create a set with increasing gaps
        # Start with a base and add increasing increments
        base = random.randint(10, 50)
        n_target = random.randint(30, 35)
        
        # Generate a set with geometric-like spacing
        elements = []
        current = base
        for i in range(n_target):
            if current > 256:
                break
            elements.append(current)
            # Increase gap: start with small gap, increase
            gap = max(1, int(1 + i * 0.3 + random.uniform(-0.2, 0.2)))
            current += gap
        
        if len(elements) < 30:
            continue
        
        # Ensure distinctness
        elements = sorted(set(elements))
        if len(elements) < 30:
            continue
        
        # Check if valid
        if check_valid(elements):
            return elements
        
        # Try to fix by local search
        current_set = elements[:]
        for i in range(200):
            if time.time() > deadline:
                break
            
            idx = random.randint(0, len(current_set)-1)
            # Try to adjust to improve separation
            if idx < len(current_set) // 2:
                delta = random.choice([1, 2, 3, 4])
            else:
                delta = random.choice([-1, -2, -3, -4])
            
            new_val = current_set[idx] + delta
            if 1 <= new_val <= 256 and new_val not in current_set:
                old_val = current_set[idx]
                current_set[idx] = new_val
                if check_valid(current_set):
                    return current_set
                current_set[idx] = old_val
    
    # Approach 2: Systematic search for size 32
    # Use the fact that for AP [a, a+1, ..., a+n-1], we need:
    # min_sum(k+1) > max_sum(k) for all k
    # This means: (k+1)*a + k(k+1)/2 > k*a + k(2a+k-1)/2 ... wait, let me recalculate
    
    # For sorted set s[0] < s[1] < ... < s[n-1]:
    # min_sum(k) = sum of k smallest = s[0] + s[1] + ... + s[k-1]
    # max_sum(k) = sum of k largest = s[n-k] + s[n-k+1] + ... + s[n-1]
    # We need min_sum(k+1) > max_sum(k) for all k = 1, ..., n-1
    
    # For AP with first term a and n terms: s[i] = a + i
    # min_sum(k+1) = (k+1)*a + k(k+1)/2
    # max_sum(k) = k*a + k(n-1) - k(k-1)/2
    # Need: (k+1)*a + k(k+1)/2 > k*a + k(n-1) - k(k-1)/2
    # => a > k(n-1) - k(k-1)/2 - k(k+1)/2
    # => a > k(n-1) - k^2
    # => a > k(n-1-k)
    # For k from 1 to n-1, the max of k(n-1-k) is at k = (n-1)/2
    # For n=32: max at k=15.5, so check k=15 and k=16
    # k=15: 15*(31-15) = 15*16 = 240
    # k=16: 16*(31-16) = 16*15 = 240
    # So a > 240, i.e., a >= 241
    # But then s[31] = 241+31 = 272 > 256. So AP of size 32 doesn't work.
    
    # Need a non-AP set. Let's try to construct one.
    # Key: we need the small elements to be large enough relative to large elements.
    
    # Try: use elements that grow faster than linear
    # e.g., s[i] = a + b*i + c*i^2 for small c
    
    for c in [1, 2, 3, 4, 5]:
        for b in range(1, 20):
            for a in range(1, 100):
                if time.time() > deadline:
                    break
                
                elements = []
                for i in range(32):
                    val = a + b*i + c*i*i
                    if val > 256:
                        break
                    elements.append(val)
                
                if len(elements) < 32:
                    continue
                
                elements = sorted(set(elements))
                if len(elements) < 32:
                    continue
                
                if check_valid(elements):
                    return elements
    
    # Approach 3: Greedy construction
    # Build set one element at a time, choosing the smallest valid next element
    for start_val in range(1, 50):
        if time.time() > deadline:
            break
        
        elements = [start_val]
        for i in range(31):
            # Try to add the smallest possible next element
            found = False
            for candidate in range(elements[-1] + 1, 257):
                test_set = elements + [candidate]
                if check_valid(test_set):
                    elements.append(candidate)
                    found = True
                    break
            
            if not found:
                break
        
        if len(elements) >= 32:
            return elements[:32]
    
    # Return fallback
    return fallback
