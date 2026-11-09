import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    def is_valid(seq):
        if not seq:
            return True
        if not all(2 <= x <= 128 for x in seq):
            return False
        if not all(seq[i] < seq[i+1] for i in range(len(seq)-1)):
            return False
        products = set()
        for i in range(len(seq)):
            prod = 1
            for j in range(i, len(seq)):
                prod *= seq[j]
                if prod in products:
                    return False
                products.add(prod)
        return True
    
    def can_add(seq, prods, x):
        if x in prods:
            return False
        suffix_prod = 1
        for i in range(len(seq) - 1, -1, -1):
            suffix_prod *= seq[i]
            new_prod = x * suffix_prod
            if new_prod in prods:
                return False
        return True
    
    def add_to_seq(seq, prods, x):
        seq.append(x)
        prods.add(x)
        suffix_prod = x
        for i in range(len(seq) - 2, -1, -1):
            suffix_prod *= seq[i]
            prods.add(suffix_prod)
    
    def deterministic_greedy():
        seq = []
        prods = set()
        for x in range(2, 129):
            if can_add(seq, prods, x):
                add_to_seq(seq, prods, x)
        return seq
    
    def randomized_greedy(skip_prob_range=(0.0, 0.4)):
        seq = []
        prods = set()
        skip_prob = random.uniform(*skip_prob_range)
        for x in range(2, 129):
            if not can_add(seq, prods, x):
                continue
            if random.random() < skip_prob:
                continue
            add_to_seq(seq, prods, x)
        return seq
    
    def local_search_improve(seq, max_iters=200):
        best = seq[:]
        best_len = len(seq)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            improved = False
            
            for i in range(len(seq)):
                if time.time() > deadline:
                    break
                
                # Remove element i and rebuild greedily with randomization
                new_seq = seq[:i] + seq[i+1:]
                new_prods = set()
                for a in range(len(new_seq)):
                    prod = 1
                    for b in range(a, len(new_seq)):
                        prod *= new_seq[b]
                        new_prods.add(prod)
                
                # Randomized greedy rebuild from position i
                test_seq = new_seq[:]
                test_prods = new_prods.copy()
                skip_prob = random.uniform(0.0, 0.3)
                
                for x in range(new_seq[i-1] + 1 if i > 0 else 2, 129):
                    if not can_add(test_seq, test_prods, x):
                        continue
                    if random.random() < skip_prob:
                        continue
                    add_to_seq(test_seq, test_prods, x)
                
                if len(test_seq) > best_len:
                    best = test_seq[:]
                    best_len = len(test_seq)
                    improved = True
            
            if not improved:
                break
        
        return best
    
    best_seq = []
    best_len = 0
    
    # Deterministic baseline
    det_seq = deterministic_greedy()
    if len(det_seq) > best_len:
        best_len = len(det_seq)
        best_seq = det_seq[:]
    
    # Randomized multi-start
    num_restarts = 0
    while time.time() < deadline - 10.0:  # Reserve 10s for local search
        seq = randomized_greedy()
        
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq[:]
        
        num_restarts += 1
    
    # Local search on best
    if best_seq and time.time() < deadline:
        improved = local_search_improve(best_seq, max_iters=500)
        if len(improved) > best_len:
            best_len = len(improved)
            best_seq = improved
    
    # Final verification
    if not is_valid(best_seq):
        def is_prime(x):
            if x < 2:
                return False
            for d in range(2, int(x**0.5) + 1):
                if x % d == 0:
                    return False
            return True
        best_seq = [x for x in range(2, 129) if is_prime(x)]
    
    return best_seq

if __name__ == "__main__":
    res = construct()
    print(f"Length: {len(res)}")
    print(res)
