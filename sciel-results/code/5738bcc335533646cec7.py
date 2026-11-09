"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
from fractions import Fraction
import time

def construct():
    # Generate primes up to 1000
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    primes = [p for p in range(2, 1001) if is_prime(p)]
    MAX_L = 2000  # Track coverage up to this limit for optimization
    BUDGET = Fraction(1, 1)
    
    best_classes = []
    best_L = 0
    
    def evaluate_L(cover_count):
        L = 0
        for i in range(1, MAX_L + 1):
            if cover_count[i] > 0:
                L = i
            else:
                break
        return L

    def add_class(p, r, cover_count, used_primes, total_weight):
        # Add (p, r) to cover_count
        start = r
        if start < 1:
            start = 1 + ((1 - r) % p) if r != 0 else 1 # Actually r is 0..p-1.
            # If r=0, numbers are p, 2p...
            # If r>0, numbers are r, r+p...
            # We want x in [1, MAX_L] such that x % p == r.
            # Smallest such x:
            if r == 0:
                start = p
            else:
                start = r
                if start < 1: start = 1 # r is 0..p-1 so r>=1 if r!=0.
        # Wait, r is 0..p-1.
        # If r=0, x is multiple of p. Smallest x>=1 is p.
        # If r>0, x=r, r+p... Smallest x>=1 is r (since r>=1).
        
        if r == 0:
            start = p
        else:
            start = r
            
        for x in range(start, MAX_L + 1, p):
            cover_count[x] += 1
        used_primes.add(p)
        total_weight += Fraction(1, p)
        return total_weight

    def remove_class(p, r, cover_count, used_primes, total_weight):
        if r == 0:
            start = p
        else:
            start = r
            
        for x in range(start, MAX_L + 1, p):
            cover_count[x] -= 1
        used_primes.remove(p)
        total_weight -= Fraction(1, p)
        return total_weight

    def get_L(cover_count):
        L = 0
        for i in range(1, MAX_L + 1):
            if cover_count[i] > 0:
                L = i
            else:
                break
        return L

    # Local Search
    random.seed(42) # For reproducibility in testing, but we want diversity? 
    # The task says "Maximize". A deterministic good heuristic + local search is best.
    
    # Initialize with a heuristic: Greedy by smallest primes, but randomized residues?
    # Let's try a few random restarts of Hill Climbing.
    
    start_time = time.time()
    time_limit = 100.0 # Leave 20s for safety
    
    for restart in range(10):
        # Initialize state
        used_primes = set()
        residues = {} # p -> r
        cover_count = [0] * (MAX_L + 1)
        total_weight = Fraction(0, 1)
        
        # Heuristic Initialization: Add smallest primes with residue 1 (or 0)
        # This gives a decent base.
        for p in primes[:10]:
            if total_weight + Fraction(1, p) <= BUDGET:
                r = 1 if p > 1 else 0 # p=2, r=1 covers odd. p=3, r=1 covers 1,4,7...
                # Actually, let's pick r randomly from 0..p-1
                r = random.randint(0, p-1)
                add_class(p, r, cover_count, used_primes, total_weight)
                residues[p] = r
        
        current_L = get_L(cover_count)
        if current_L > best_L:
            best_L = current_L
            best_classes = [[p, residues[p]] for p in used_primes]
            
        # Hill Climbing
        no_improve_count = 0
        max_no_improve = 500
        
        while time.time() - start_time < time_limit and no_improve_count < max_no_improve:
            # Pick a random move
            move_type = random.choice(['change_res', 'add', 'remove', 'swap'])
            
            accepted = False
            old_L = current_L
            
            if move_type == 'change_res' and used_primes:
                p = random.choice(list(used_primes))
                old_r = residues[p]
                new_r = random.randint(0, p-1)
                if new_r == old_r:
                    continue
                
                # Remove old, add new
                remove_class(p, old_r, cover_count, used_primes, total_weight)
                add_class(p, new_r, cover_count, used_primes, total_weight)
                residues[p] = new_r
                
                new_L = get_L(cover_count)
                
                if new_L > old_L:
                    current_L = new_L
                    no_improve_count = 0
                    if current_L > best_L:
                        best_L = current_L
                        best_classes = [[p, residues[p]] for p in used_primes]
                else:
                    # Revert
                    remove_class(p, new_r, cover_count, used_primes, total_weight)
                    add_class(p, old_r, cover_count, used_primes, total_weight)
                    residues[p] = old_r
                    no_improve_count += 1
                    
            elif move_type == 'add':
                available = [p for p in primes if p not in used_primes]
                if not available:
                    continue
                p = random.choice(available)
                if total_weight + Fraction(1, p) > BUDGET:
                    continue
                r = random.randint(0, p-1)
                
                add_class(p, r, cover_count, used_primes, total_weight)
                residues[p] = r
                
                new_L = get_L(cover_count)
                
                if new_L > old_L:
                    current_L = new_L
                    no_improve_count = 0
                    if current_L > best_L:
                        best_L = current_L
                        best_classes = [[p, residues[p]] for p in used_primes]
                else:
                    # Revert
                    remove_class(p, r, cover_count, used_primes, total_weight)
                    del residues[p]
                    no_improve_count += 1

            elif move_type == 'remove' and used_primes:
                p = random.choice(list(used_primes))
                r = residues[p]
                
                remove_class(p, r, cover_count, used_primes, total_weight)
                del residues[p]
                
                new_L = get_L(cover_count)
                
                if new_L > old_L:
                    current_L = new_L
                    no_improve_count = 0
                    if current_L > best_L:
                        best_L = current_L
                        best_classes = [[p, residues[p]] for p in used_primes]
                else:
                    # Revert
                    add_class(p, r, cover_count, used_primes, total_weight)
                    residues[p] = r
                    no_improve_count += 1
                    
            elif move_type == 'swap' and used_primes:
                # Swap one used prime with one unused prime
                available = [p for p in primes if p not in used_primes]
                if not available:
                    continue
                p_out = random.choice(list(used_primes))
                p_in = random.choice(available)
                
                # Check budget
                new_weight = total_weight - Fraction(1, p_out) + Fraction(1, p_in)
                if new_weight > BUDGET:
                    continue
                    
                r_out = residues[p_out]
                r_in = random.randint(0, p_in-1)
                
                # Apply swap
                remove_class(p_out, r_out, cover_count, used_primes, total_weight)
                del residues[p_out]
                add_class(p_in, r_in, cover_count, used_primes, total_weight)
                residues[p_in] = r_in
                
                new_L = get_L(cover_count)
                
                if new_L > old_L:
                    current_L = new_L
                    no_improve_count = 0
                    if current_L > best_L:
                        best_L = current_L
                        best_classes = [[p, residues[p]] for p in used_primes]
                else:
                    # Revert
                    remove_class(p_in, r_in, cover_count, used_primes, total_weight)
                    del residues[p_in]
                    add_class(p_out, r_out, cover_count, used_primes, total_weight)
                    residues[p_out] = r_out
                    no_improve_count += 1
                    
        if time.time() - start_time > time_limit:
            break

    # Final verification and cleanup
    if not best_classes:
        # Fallback to a simple valid solution
        return [[2, 1]]
        
    # Verify budget
    total = sum(Fraction(1, p) for p, r in best_classes)
    if total > 1:
        # This shouldn't happen, but if it does, return a safe subset
        return [[2, 1]]
        
    return best_classes
# EVOLVE-BLOCK-END
