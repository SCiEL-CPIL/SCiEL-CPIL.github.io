"""Rewritten program with a stronger algorithm for integer sum-product extremal sets."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 75.0  # Safety margin before 90s kill
    best_set = None
    best_score = float('inf')
    
    def evaluate(A):
        """Compute max(|A+A|, |A*A|) for set A."""
        A = sorted(A)
        n = len(A)
        sumset = set()
        for i in range(n):
            for j in range(i, n):
                sumset.add(A[i] + A[j])
        prodset = set()
        for i in range(n):
            for j in range(i, n):
                prodset.add(A[i] * A[j])
        return max(len(sumset), len(prodset))
    
    def divisors(n):
        divs = set()
        i = 1
        while i * i <= n:
            if n % i == 0:
                divs.add(i)
                divs.add(n // i)
            i += 1
        return divs
    
    # Try structured initializations
    initial_sets = []
    
    # Sequential
    initial_sets.append(list(range(1, 21)))
    
    # Geometric progression with ratio 2
    initial_sets.append([2**i for i in range(20)])
    
    # Divisors of highly composite numbers
    for hc in [720, 1260, 2520, 5040, 10080, 15120, 25200, 50400]:
        divs = sorted([d for d in divisors(hc) if d <= 10000])
        if len(divs) >= 20:
            initial_sets.append(divs[:20])
    
    # Lattice-like: products of small primes
    for p, q in [(2,3), (2,5), (3,5), (2,7), (3,7)]:
        lattice = set()
        for a in range(5):
            for b in range(5):
                val = (p**a) * (q**b)
                if val <= 10000:
                    lattice.add(val)
        if len(lattice) >= 20:
            initial_sets.append(sorted(lattice)[:20])
    
    # Mixed: small numbers + some structured additions
    initial_sets.append(list(range(1, 11)) + [12, 15, 18, 20, 24, 30, 36, 40, 45, 48])
    
    # Additional structured sets
    # Powers of 2 mixed with small multiples
    initial_sets.append([1,2,3,4,6,8,12,16,24,32,48,64,96,128,192,256,384,512,768,1024])
    # Fibonacci-like
    initial_sets.append([1,2,3,5,8,13,21,34,55,89,144,233,377,610,987,1597,2584,4181,6765,10946])
    # Small primes + composites
    initial_sets.append([2,3,5,7,11,13,17,19,23,29,4,6,8,9,10,12,14,15,16,18])
    
    # Evaluate all initial sets
    for A in initial_sets:
        if len(A) != 20 or len(set(A)) != 20:
            continue
        if any(x < 1 or x > 10000 for x in A):
            continue
        score = evaluate(A)
        if score < best_score:
            best_score = score
            best_set = sorted(A)[:]
    
    if best_set is None:
        best_set = list(range(1, 21))
        best_score = evaluate(best_set)
    
    if time.time() > deadline:
        return best_set
    
    # Incremental evaluation class
    class IncrementalEval:
        def __init__(self, A):
            self.A = sorted(A)
            self.n = len(self.A)
            self.sumset = set()
            self.prodset = set()
            for i in range(self.n):
                for j in range(i, self.n):
                    self.sumset.add(self.A[i] + self.A[j])
                    self.prodset.add(self.A[i] * self.A[j])
            self.sum_counts = {}
            self.prod_counts = {}
            for s in self.sumset:
                self.sum_counts[s] = 0
            for p in self.prodset:
                self.prod_counts[p] = 0
            for i in range(self.n):
                for j in range(i, self.n):
                    s = self.A[i] + self.A[j]
                    self.sum_counts[s] += 1
                    p = self.A[i] * self.A[j]
                    self.prod_counts[p] += 1
        
        def get_score(self):
            return max(len(self.sumset), len(self.prodset))
        
        def replace_element(self, idx, new_val):
            """Replace element at idx with new_val, updating sets incrementally."""
            old_val = self.A[idx]
            if old_val == new_val:
                return
            
            # Remove all pairs involving old_val
            for i in range(self.n):
                if i == idx:
                    continue
                s = old_val + self.A[i]
                p = old_val * self.A[i]
                self.sum_counts[s] -= 1
                if self.sum_counts[s] == 0:
                    self.sumset.discard(s)
                    del self.sum_counts[s]
                self.prod_counts[p] -= 1
                if self.prod_counts[p] == 0:
                    self.prodset.discard(p)
                    del self.prod_counts[p]
            # Remove diagonal pair (old_val, old_val)
            s_diag = old_val + old_val
            p_diag = old_val * old_val
            self.sum_counts[s_diag] -= 1
            if self.sum_counts[s_diag] == 0:
                self.sumset.discard(s_diag)
                del self.sum_counts[s_diag]
            self.prod_counts[p_diag] -= 1
            if self.prod_counts[p_diag] == 0:
                self.prodset.discard(p_diag)
                del self.prod_counts[p_diag]
            
            # Add new element
            self.A[idx] = new_val
            self.A.sort()
            idx = self.A.index(new_val)
            
            # Add all pairs involving new_val
            for i in range(self.n):
                if i == idx:
                    continue
                s = new_val + self.A[i]
                p = new_val * self.A[i]
                if s not in self.sum_counts:
                    self.sumset.add(s)
                    self.sum_counts[s] = 0
                self.sum_counts[s] += 1
                if p not in self.prod_counts:
                    self.prodset.add(p)
                    self.prod_counts[p] = 0
                self.prod_counts[p] += 1
            # Add diagonal pair
            s_diag = new_val + new_val
            p_diag = new_val * new_val
            if s_diag not in self.sum_counts:
                self.sumset.add(s_diag)
                self.sum_counts[s_diag] = 0
            self.sum_counts[s_diag] += 1
            if p_diag not in self.prod_counts:
                self.prodset.add(p_diag)
                self.prod_counts[p_diag] = 0
            self.prod_counts[p_diag] += 1
        
        def replace_two_elements(self, idx1, val1, idx2, val2):
            """Replace two elements simultaneously."""
            old1 = self.A[idx1]
            old2 = self.A[idx2]
            
            # Remove all pairs involving old1 or old2
            for i in range(self.n):
                if i == idx1 or i == idx2:
                    continue
                # Pairs with old1
                s = old1 + self.A[i]
                p = old1 * self.A[i]
                self.sum_counts[s] -= 1
                if self.sum_counts[s] == 0:
                    self.sumset.discard(s)
                    del self.sum_counts[s]
                self.prod_counts[p] -= 1
                if self.prod_counts[p] == 0:
                    self.prodset.discard(p)
                    del self.prod_counts[p]
                # Pairs with old2
                s = old2 + self.A[i]
                p = old2 * self.A[i]
                self.sum_counts[s] -= 1
                if self.sum_counts[s] == 0:
                    self.sumset.discard(s)
                    del self.sum_counts[s]
                self.prod_counts[p] -= 1
                if self.prod_counts[p] == 0:
                    self.prodset.discard(p)
                    del self.prod_counts[p]
            
            # Remove diagonal pairs
            for old in [old1, old2]:
                s_diag = old + old
                p_diag = old * old
                self.sum_counts[s_diag] -= 1
                if self.sum_counts[s_diag] == 0:
                    self.sumset.discard(s_diag)
                    del self.sum_counts[s_diag]
                self.prod_counts[p_diag] -= 1
                if self.prod_counts[p_diag] == 0:
                    self.prodset.discard(p_diag)
                    del self.prod_counts[p_diag]
            
            # Remove pair between old1 and old2 (if different)
            if old1 != old2:
                s = old1 + old2
                p = old1 * old2
                self.sum_counts[s] -= 1
                if self.sum_counts[s] == 0:
                    self.sumset.discard(s)
                    del self.sum_counts[s]
                self.prod_counts[p] -= 1
                if self.prod_counts[p] == 0:
                    self.prodset.discard(p)
                    del self.prod_counts[p]
            
            # Update array
            self.A[idx1] = val1
            self.A[idx2] = val2
            self.A.sort()
            idx1 = self.A.index(val1)
            idx2 = self.A.index(val2)
            
            # Add all pairs involving val1 or val2
            for i in range(self.n):
                if i == idx1 or i == idx2:
                    continue
                # Pairs with val1
                s = val1 + self.A[i]
                p = val1 * self.A[i]
                if s not in self.sum_counts:
                    self.sumset.add(s)
                    self.sum_counts[s] = 0
                self.sum_counts[s] += 1
                if p not in self.prod_counts:
                    self.prodset.add(p)
                    self.prod_counts[p] = 0
                self.prod_counts[p] += 1
                # Pairs with val2
                s = val2 + self.A[i]
                p = val2 * self.A[i]
                if s not in self.sum_counts:
                    self.sumset.add(s)
                    self.sum_counts[s] = 0
                self.sum_counts[s] += 1
                if p not in self.prod_counts:
                    self.prodset.add(p)
                    self.prod_counts[p] = 0
                self.prod_counts[p] += 1
            
            # Add diagonal pairs
            for new in [val1, val2]:
                s_diag = new + new
                p_diag = new * new
                if s_diag not in self.sum_counts:
                    self.sumset.add(s_diag)
                    self.sum_counts[s_diag] = 0
                self.sum_counts[s_diag] += 1
                if p_diag not in self.prod_counts:
                    self.prodset.add(p_diag)
                    self.prod_counts[p_diag] = 0
                self.prod_counts[p_diag] += 1
            
            # Add pair between val1 and val2 (if different)
            if val1 != val2:
                s = val1 + val2
                p = val1 * val2
                if s not in self.sum_counts:
                    self.sumset.add(s)
                    self.sum_counts[s] = 0
                self.sum_counts[s] += 1
                if p not in self.prod_counts:
                    self.prodset.add(p)
                    self.prod_counts[p] = 0
                self.prod_counts[p] += 1
    
    # Simulated annealing with incremental evaluation
    random.seed(42)
    current = best_set[:]
    current_score = best_score
    temperature = 50.0
    cooling = 0.9999
    
    # Use incremental evaluation
    current_A = sorted(current)
    inc_eval = IncrementalEval(current_A)
    
    for iteration in range(500000):
        if time.time() > deadline:
            break
        
        move_type = random.random()
        
        if move_type < 0.7:
            # Nudge: change one element by a small amount
            i = random.randint(0, 19)
            old_val = current_A[i]
            # Try different nudge sizes
            if random.random() < 0.5:
                nudge = random.randint(-3, 3)
            else:
                nudge = random.randint(-10, 10)
            new_val = old_val + nudge
            if new_val < 1 or new_val > 10000:
                continue
            if new_val in current_A:
                continue
            
            # Apply move
            inc_eval.replace_element(i, new_val)
            new_score = inc_eval.get_score()
            delta = new_score - current_score
            
            if delta < 0 or random.random() < (1.0 / (1.0 + 0.5 * abs(delta))) * min(1.0, temperature / 10.0):
                current_score = new_score
                current_A = inc_eval.A[:]
                if new_score < best_score:
                    best_score = new_score
                    best_set = current_A[:]
            else:
                # Revert
                inc_eval.replace_element(i, old_val)
                current_A = inc_eval.A[:]
        elif move_type < 0.95:
            # Swap: change two elements
            i, j = random.sample(range(20), 2)
            old_i, old_j = current_A[i], current_A[j]
            nudge_i = random.randint(-5, 5)
            nudge_j = random.randint(-5, 5)
            new_i = old_i + nudge_i
            new_j = old_j + nudge_j
            if new_i < 1 or new_i > 10000 or new_j < 1 or new_j > 10000:
                continue
            if new_i in current_A or new_j in current_A:
                continue
            if new_i == new_j:
                continue
            
            # Apply move
            inc_eval.replace_two_elements(i, new_i, j, new_j)
            new_score = inc_eval.get_score()
            delta = new_score - current_score
            
            if delta < 0 or random.random() < (1.0 / (1.0 + 0.5 * abs(delta))) * min(1.0, temperature / 10.0):
                current_score = new_score
                current_A = inc_eval.A[:]
                if new_score < best_score:
                    best_score = new_score
                    best_set = current_A[:]
            else:
                # Revert
                inc_eval.replace_two_elements(i, old_i, j, old_j)
                current_A = inc_eval.A[:]
        else:
            # Large jump: replace one element with a random value
            i = random.randint(0, 19)
            old_val = current_A[i]
            new_val = random.randint(1, 10000)
            if new_val in current_A:
                continue
            
            # Apply move
            inc_eval.replace_element(i, new_val)
            new_score = inc_eval.get_score()
            delta = new_score - current_score
            
            if delta < 0 or random.random() < (1.0 / (1.0 + 0.5 * abs(delta))) * min(1.0, temperature / 10.0):
                current_score = new_score
                current_A = inc_eval.A[:]
                if new_score < best_score:
                    best_score = new_score
                    best_set = current_A[:]
            else:
                # Revert
                inc_eval.replace_element(i, old_val)
                current_A = inc_eval.A[:]
        
        temperature *= cooling
        if temperature < 0.1:
            temperature = 0.1
    
    # Final polish pass: try all single-element nudges with small steps
    if time.time() < deadline - 5:
        polish_A = best_set[:]
        polish_score = best_score
        polish_eval = IncrementalEval(polish_A)
        
        for _ in range(100):
            improved = False
            for i in range(20):
                old_val = polish_A[i]
                for nudge in range(-10, 11):
                    if nudge == 0:
                        continue
                    new_val = old_val + nudge
                    if new_val < 1 or new_val > 10000:
                        continue
                    if new_val in polish_A:
                        continue
                    
                    polish_eval.replace_element(i, new_val)
                    new_score = polish_eval.get_score()
                    
                    if new_score < polish_score:
                        polish_score = new_score
                        polish_A = polish_eval.A[:]
                        improved = True
                        break
                    else:
                        polish_eval.replace_element(i, old_val)
                        polish_A = polish_eval.A[:]
                if improved:
                    break
            if not improved:
                break
        
        if polish_score < best_score:
            best_score = polish_score
            best_set = polish_A[:]
    
    return best_set
# EVOLVE-BLOCK-END
