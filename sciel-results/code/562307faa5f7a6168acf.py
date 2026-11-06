import time
import random
import sys
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Precompute signatures for numbers 2..128
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    P = len(primes)
    
    def get_sig(x):
        sig = [0] * P
        n = x
        for idx, p in enumerate(primes):
            if p * p > n:
                break
            while n % p == 0:
                sig[idx] += 1
                n //= p
        if n > 1:
            idx = prime_to_idx[n]
            sig[idx] += 1
        return tuple(sig)
    
    sigs = [None] * 129
    for i in range(2, 129):
        sigs[i] = get_sig(i)
        
    # Precompute all pairwise products as signatures
    # For a sequence, the product of consecutive blocks from position i to j is the sum of sigs[i] through sigs[j]
    # We need to ensure all such products are distinct
    
    # Use a more efficient approach: maintain suffix products incrementally
    # When we add a new element x at the end, the new products are:
    # x, x * last, x * last * second_last, etc.
    # We can maintain a list of "active" product signatures
    
    def run_greedy():
        seq = []
        products_set = set()
        suffix_sigs = []  # suffix_sigs[k] = product of seq[-k:] for k=1..len(seq)
        
        for x in range(2, 129):
            x_sig = sigs[x]
            # Compute new tails: x, x*last, x*last*second_last, ...
            new_tails = []
            current = x_sig
            new_tails.append(current)
            for s in reversed(suffix_sigs):
                current = tuple(a + b for a, b in zip(current, s))
                new_tails.append(current)
            
            # Check for collisions
            valid = all(t not in products_set for t in new_tails)
            
            if valid:
                seq.append(x)
                products_set.update(new_tails)
                suffix_sigs.append(x_sig)
                
        return seq
    
    # Get initial solution
    best_seq = run_greedy()
    best_len = len(best_seq)
    
    # Now try to improve using local search / hill climbing
    # Strategy: try removing and re-adding elements, or swapping
    
    def get_seq_signature(seq):
        """Compute all consecutive block products for a sequence"""
        products = set()
        for i in range(len(seq)):
            current = [0] * P
            for j in range(i, len(seq)):
                for k in range(P):
                    current[k] += sigs[seq[j]][k]
                prod_sig = tuple(current)
                if prod_sig in products:
                    return None  # Invalid
                products.add(prod_sig)
        return len(seq)
    
    # Local search: try to improve by modifying the sequence
    def local_search(start_seq, max_iterations=10000):
        current_seq = start_seq[:]
        current_len = len(current_seq)
        best_local = current_seq[:]
        best_local_len = current_len
        
        for iter in range(max_iterations):
            if time.time() > deadline - 5.0:
                break
            
            # Try different moves
            move_type = random.random()
            
            if move_type < 0.3:
                # Try adding a missing element at a random position
                missing = [x for x in range(2, 129) if x not in current_seq]
                if not missing:
                    continue
                x = random.choice(missing)
                pos = random.randint(0, len(current_seq))
                new_seq = current_seq[:pos] + [x] + current_seq[pos:]
                # Check if still strictly increasing
                if all(new_seq[i] < new_seq[i+1] for i in range(len(new_seq)-1)):
                    if get_seq_signature(new_seq) is not None:
                        if len(new_seq) > best_local_len:
                            best_local = new_seq[:]
                            best_local_len = len(new_seq)
                        current_seq = new_seq
            elif move_type < 0.6:
                # Try removing an element
                if len(current_seq) > 2:
                    pos = random.randint(0, len(current_seq)-1)
                    new_seq = current_seq[:pos] + current_seq[pos+1:]
                    if len(new_seq) > 0:
                        if get_seq_signature(new_seq) is not None:
                            current_seq = new_seq
            else:
                # Try swapping two adjacent elements
                if len(current_seq) >= 2:
                    pos = random.randint(0, len(current_seq)-2)
                    if current_seq[pos] + 1 == current_seq[pos+1]:
                        # Can't swap if they're consecutive in value (would break increasing order)
                        pass
                    else:
                        new_seq = current_seq[:]
                        new_seq[pos], new_seq[pos+1] = new_seq[pos+1], new_seq[pos]
                        if all(new_seq[i] < new_seq[i+1] for i in range(len(new_seq)-1)):
                            if get_seq_signature(new_seq) is not None:
                                current_seq = new_seq
        
        return best_local, best_local_len
    
    # Run local search from the greedy solution
    improved_seq, improved_len = local_search(best_seq, max_iterations=5000)
    if improved_len > best_len:
        best_seq = improved_seq
        best_len = improved_len
    
    # Try multiple random restarts with perturbation
    for restart in range(50):
        if time.time() > deadline - 10.0:
            break
        
        # Perturb the initial greedy solution
        perturbed = best_seq[:]
        # Randomly remove some elements
        for _ in range(random.randint(1, 5)):
            if len(perturbed) > 2:
                pos = random.randint(0, len(perturbed)-1)
                perturbed.pop(pos)
        
        # Run local search from perturbed state
        improved_seq, improved_len = local_search(perturbed, max_iterations=2000)
        if improved_len > best_len:
            best_seq = improved_seq
            best_len = improved_len
    
    return best_seq
