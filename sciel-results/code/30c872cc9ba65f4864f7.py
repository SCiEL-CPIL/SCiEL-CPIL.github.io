"""
Strategy: Simulated Annealing for Prime Residue Class Optimization.

The previous greedy approach commits to a prime-residue pair early, potentially
preventing a globally better configuration. This strategy uses Simulated Annealing
to explore the space of possible sets of prime-residue classes.

State Representation:
- A set of active primes (subset of primes <= 1000).
- For each active prime, a chosen residue r in [0, p-1].

Objective Function:
- Maximize L (largest integer such that 1..L is covered).
- Constraint: Sum of 1/p for active primes <= 1.

Move Operators:
1. Add a new prime (random from inactive, random residue).
2. Remove an existing prime.
3. Change the residue of an existing prime.
4. Swap: Remove one prime, add another.

Acceptance:
- Metropolis criterion: Accept if score improves, or with probability exp((new_score - old_score) / T) if it worsens.
- Score definition: We want to maximize L. However, L is a step function.
  To make the landscape smoother, we can define a score that rewards coverage density
  near the frontier and penalizes budget violations.
  
  Actually, simpler: 
  Score = L if budget is valid, else -infinity (or a large penalty).
  But for SA, we need a continuous-ish landscape. 
  Let's use: 
  Base Score = L.
  If budget > 1, apply a heavy penalty proportional to the excess.
  
  Even better for SA: 
  We want to find a valid configuration with max L.
  Let's define energy E = -L + penalty.
  Minimize E.
  
  Penalty for budget violation: K * (sum(1/p) - 1) if sum > 1, else 0.
  We can also add a small penalty for unused budget to encourage tight packing? 
  No, we just need sum <= 1.
  
  To handle the discrete nature of L, we can also add a "frontier pressure" term:
  Count how many numbers in [L+1, L+window] are covered. This helps guide the search 
  to extend the frontier.
  
  Let's define:
  valid = sum(1/p) <= 1
  if not valid:
      energy = 1000000 + (sum(1/p) - 1) * 1000  # Large penalty
  else:
      # Calculate L
      # To speed up, we don't recalculate L from scratch every time if possible?
      # But L depends on all residues. 
      # Optimization: Keep a coverage array for a fixed range (e.g., 1..5000).
      # When a move is made, update the coverage array incrementally.
      # Then L is the first gap.
      
  Let's implement incremental coverage updates.
  Range: We care about L up to ~2000-5000. Let's set MAX_L = 5000.
  Coverage array: boolean array of size MAX_L+1.
  
  Moves:
  - Add prime p with residue r: 
    - For all n in [1, MAX_L] where n % p == r, set covered[n] = True.
    - Update budget.
  - Remove prime p with residue r:
    - For all n in [1, MAX_L] where n % p == r, we CANNOT simply set covered[n] = False 
      because another prime might cover it.
    - This makes incremental removal hard.
    
  Alternative: 
  Since the number of primes is small (<= 168) and we can only afford ~10-15 primes 
  (sum 1/p <= 1), the set of active primes is small.
  Recalculating L from scratch is O(MAX_L * num_active_primes).
  If MAX_L = 5000 and num_active = 15, that's 75,000 operations.
  With 10,000 SA steps, that's 750M operations. Might be too slow in Python.
  
  Let's optimize L calculation:
  We can check coverage for n=1..MAX_L using a bitset or vectorized numpy?
  NumPy is available.
  
  Using NumPy:
  covered = np.zeros(MAX_L+1, dtype=bool)
  For each (p, r) in active:
      covered[r::p] = True  # Vectorized slice assignment? 
      # Wait, covered[r::p] = True works if covered is a numpy array.
      # This is very fast in C under the hood.
  
  Then L = np.argmax(~covered[1:]) - 1? 
  No, np.argmax returns first True. We want first False in ~covered.
  gap_indices = np.where(~covered[1:])[0]
  if len(gap_indices) > 0:
      L = gap_indices[0]  # This is 1-based index in the slice, so actual number is gap_indices[0] + 1?
      # covered[1:] corresponds to numbers 1..MAX_L.
      # If covered[1] is False, ~covered[1:] is True at index 0. 
      # So L = 0.
      # If covered[1], covered[2] are True, covered[3] is False.
      # ~covered[1:] is [F, F, T, ...]. argmax is 2.
      # Number is 3. So L = 2.
      # Formula: L = gap_indices[0] if we treat 0-based index in slice as "number of covered items before gap"?
      # Slice index 0 -> number 1.
      # Slice index k -> number k+1.
      # If gap at slice index k, then number k+1 is not covered.
      # So L = k.
      # Yes.
  else:
      L = MAX_L

  This NumPy approach should be fast enough for ~10k-50k steps.
"""

import time
import random
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    # Primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    primes = sieve(1000)
    num_primes = len(primes)
    
    # Search space limit
    MAX_L = 5000  # We'll try to cover up to this. If we exceed, we can increase, but 5000 is likely safe for budget 1.
    
    # Initial state: Empty set
    # active_primes: list of indices into `primes`
    # residues: dict {prime_index: residue}
    
    # Better representation for fast NumPy updates:
    # active_classes: list of (p, r)
    
    def calculate_L(active_classes):
        if not active_classes:
            return 0
        covered = np.zeros(MAX_L + 1, dtype=bool)
        for p, r in active_classes:
            # r is in 0..p-1. 
            # Numbers covered: r, r+p, ...
            # If r == 0, covers p, 2p, ... (and 0, but we start at 1)
            # np slice: start, stop, step
            # We want indices i in [1, MAX_L] such that i % p == r.
            # First such index >= 1:
            # if r == 0: first is p (if p<=MAX_L)
            # if r > 0: first is r (if r<=MAX_L)
            
            # Using slice: 
            # indices = np.arange(1, MAX_L+1)
            # mask = (indices % p) == r
            # This is O(MAX_L) per prime. 
            # With 15 primes, 15*5000 = 75k ops. 
            # In NumPy, this is very fast.
            
            # Alternatively, use step slicing on the boolean array?
            # covered[r::p] = True
            # This sets indices r, r+p, r+2p...
            # If r=0, it sets 0, p, 2p...
            # We only care about 1..MAX_L. Index 0 is ignored in L calc.
            # So covered[r::p] = True is correct and faster.
            if r <= MAX_L:
                covered[r::p] = True
            # If r > MAX_L, no coverage in range.
            
        # Find first gap in 1..MAX_L
        # covered[1:] is the array for 1..MAX_L
        # We want the first index in covered[1:] that is False.
        # ~covered[1:] is True where covered is False.
        # np.argmax returns first True.
        # If all True, argmax is 0? No, if all False (in ~covered), argmax is 0.
        # If ~covered[1:] is all False (meaning covered[1:] is all True), argmax returns 0.
        # But we need to distinguish "all covered" from "gap at 1".
        
        # Let's use np.where
        gaps = np.where(~covered[1:])[0]
        if len(gaps) > 0:
            return gaps[0]  # As derived before: gap at slice index k means number k+1 is uncovered, so L=k.
        else:
            return MAX_L

    def get_budget(active_classes):
        # Use Fraction for exactness? 
        # For SA, we need speed. Floats might be okay if we check strictly at the end?
        # But "exact rationals" is required for the final output.
        # During SA, we can use floats for speed, but ensure final check is exact.
        # Or use Fraction if it's not too slow. 15 additions of Fractions is fast.
        total = Fraction(0, 1)
        for p, r in active_classes:
            total += Fraction(1, p)
        return total

    # Initial random valid state
    # Start with a few small primes to have a non-zero L
    random.seed(42)
    
    # Strategy: Start with a greedy solution, then anneal?
    # Or start empty?
    # Let's start with a simple greedy to get a decent starting point.
    
    def greedy_init():
        classes = []
        budget = Fraction(0, 1)
        # Sort primes by 1/p descending (smallest primes first)
        # But we need to pick residues wisely.
        # For init, just pick random residues for smallest primes until budget full?
        # Better: Pick smallest primes, random residues.
        for p in primes:
            if budget + Fraction(1, p) <= Fraction(1, 1):
                r = random.randint(0, p - 1)
                classes.append((p, r))
                budget += Fraction(1, p)
            else:
                # Try to find a prime that fits? 
                # For init, just stop or skip.
                pass
        return classes

    current_classes = greedy_init()
    current_L = calculate_L(current_classes)
    current_budget = get_budget(current_classes)
    
    # Ensure valid start
    if current_budget > 1:
        # Remove until valid
        while current_budget > 1 and current_classes:
            idx = random.randint(0, len(current_classes) - 1)
            p, r = current_classes.pop(idx)
            current_budget -= Fraction(1, p)
        current_L = calculate_L(current_classes)
        
    best_classes = current_classes[:]
    best_L = current_L
    best_budget = current_budget
    
    # Annealing parameters
    T = 10.0
    cooling = 0.995
    T_min = 0.01
    
    steps = 0
    max_steps = 20000 # Adjust based on time
    
    while time.time() < deadline and T > T_min:
        steps += 1
        
        # Generate a move
        move_type = random.random()
        
        new_classes = current_classes[:]
        
        if move_type < 0.3 and len(current_classes) < 20:
            # Add a prime
            # Pick a prime not in current
            active_p_set = {p for p, r in current_classes}
            available = [p for p in primes if p not in active_p_set]
            if available:
                p = random.choice(available)
                # Prefer small primes? Random is fine for diversity.
                # But small primes are more "expensive". 
                # Let's pick uniformly from available.
                r = random.randint(0, p - 1)
                new_classes.append((p, r))
                
        elif move_type < 0.5 and current_classes:
            # Remove a prime
            idx = random.randint(0, len(current_classes) - 1)
            new_classes.pop(idx)
            
        elif move_type < 0.8 and current_classes:
            # Change residue
            idx = random.randint(0, len(current_classes) - 1)
            p, r = new_classes[idx]
            new_r = random.randint(0, p - 1)
            new_classes[idx] = (p, new_r)
            
        else:
            # Swap: Remove one, add one
            if current_classes:
                idx = random.randint(0, len(current_classes) - 1)
                p_old, r_old = new_classes.pop(idx)
                
                active_p_set = {p for p, r in new_classes}
                available = [p for p in primes if p not in active_p_set]
                if available:
                    p_new = random.choice(available)
                    r_new = random.randint(0, p_new - 1)
                    new_classes.append((p_new, r_new))
                else:
                    # Put it back if no swap possible
                    new_classes.insert(idx, (p_old, r_old))

        # Evaluate new state
        # Fast check: Budget
        new_budget = get_budget(new_classes)
        
        if new_budget > Fraction(1, 1):
            # Invalid. 
            # Energy penalty
            # Calculate L anyway for comparison? Or just reject?
            # Rejecting invalid moves immediately is standard.
            # But sometimes we need to go through invalid states?
            # Since the constraint is hard, let's just reject.
            # However, if we are stuck, we might want to allow temporary violations?
            # No, keep it simple: Hard constraint.
            continue
        
        new_L = calculate_L(new_classes)
        
        # Acceptance
        if new_L > current_L:
            current_classes = new_classes
            current_L = new_L
            current_budget = new_budget
            
            if new_L > best_L:
                best_classes = new_classes[:]
                best_L = new_L
                best_budget = new_budget
                
        else:
            # Delta = new_L - current_L (negative)
            delta = new_L - current_L
            # Probability
            if delta == 0:
                prob = 1.0
            else:
                # exp(delta / T)
                # delta is negative
                prob = np.exp(delta / T)
            
            if random.random() < prob:
                current_classes = new_classes
                current_L = new_L
                current_budget = new_budget

        T *= cooling
        
        # Check time
        if time.time() > deadline - 5:
            break

    # Final verification and cleanup
    # Ensure best_classes is valid
    final_budget = get_budget(best_classes)
    if final_budget > Fraction(1, 1):
        # This shouldn't happen if we only accept valid states, but just in case
        # Fallback to empty or simple valid set
        return []
        
    # Convert to list of lists for output
    result = [[p, r] for p, r in best_classes]
    
    return result

if __name__ == "__main__":
    classes = construct()
    L = 0
    for i in range(1, 10000):
        covered = False
        for p, r in classes:
            if i % p == r:
                covered = True
                break
        if covered:
            L = i
        else:
            break
    print(f"Covered up to L={L}")
    print(f"Classes: {classes}")
    budget = sum(Fraction(1, p) for p, r in classes)
    print(f"Budget: {budget}")
