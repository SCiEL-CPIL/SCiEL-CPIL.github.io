"""Improved program: beam search with 2-move lookahead local search."""
import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 87.0  # safety margin before 90s kill

    def get_all_products(seq):
        """Compute all consecutive block products for a sequence."""
        products = set()
        n = len(seq)
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                products.add(prod)
        return products

    def is_valid(seq):
        """Check if sequence is strictly increasing and all block products are distinct."""
        if len(seq) == 0:
            return True
        for i in range(len(seq) - 1):
            if seq[i] >= seq[i+1]:
                return False
        if any(x < 2 or x > 128 for x in seq):
            return False
        products = get_all_products(seq)
        return len(products) == len(seq) * (len(seq) + 1) // 2

    def greedy_construct():
        """Standard greedy: try numbers 2..128 in order."""
        values, products = [], set()
        for x in range(2, 129):
            tails, prod = [x], x
            for a in reversed(values):
                prod *= a
                tails.append(prod)
            if not products.intersection(tails):
                values.append(x)
                products.update(tails)
        return values

    # Beam search with value-iterative approach
    def beam_search(width=50, max_steps=127):
        """
        Beam search where each state is (sequence, set_of_products).
        At each step, we can either skip the current candidate or add it.
        We keep the top `width` states by length (and lexicographically for ties).
        """
        all_candidates = list(range(2, 129))
        
        # Initialize beam with empty sequence
        beam = [(tuple(), frozenset())]
        
        for x in all_candidates:
            if time.time() - start_time > DEADLINE:
                break
            
            new_beam = []
            for seq, prods in beam:
                # Option 1: skip x
                new_beam.append((seq, prods))
                
                # Option 2: add x if valid
                if seq and seq[-1] >= x:
                    continue  # must be strictly increasing
                
                # Compute new products
                new_prods_set = set(prods)
                prod = x
                for a in reversed(seq):
                    prod *= a
                    if prod in new_prods_set:
                        break
                    new_prods_set.add(prod)
                else:
                    # No collision
                    new_seq = seq + (x,)
                    new_beam.append((new_seq, frozenset(new_prods_set)))
            
            # Keep top `width` states by length, then lexicographically
            new_beam.sort(key=lambda s: (-len(s[0]), s[0]))
            beam = new_beam[:width]
        
        # Return the longest sequence from the beam
        best = max(beam, key=lambda s: len(s[0]))
        return list(best[0])

    # Try beam search with different widths
    best_seq = greedy_construct()
    best_len = len(best_seq)
    
    if time.time() - start_time < DEADLINE:
        bs_result = beam_search(width=30)
        if is_valid(bs_result) and len(bs_result) > best_len:
            best_seq = bs_result
            best_len = len(bs_result)
    
    if time.time() - start_time < DEADLINE:
        bs_result = beam_search(width=50)
        if is_valid(bs_result) and len(bs_result) > best_len:
            best_seq = bs_result
            best_len = len(bs_result)

    # 2-move lookahead local search: remove one element, then try to add two
    def local_search_2move(seq, remaining, max_iters=100):
        """
        Try to improve sequence by removing one element and adding two others.
        This helps escape local optima where single insertions are blocked.
        """
        current = list(seq)
        current_remaining = list(remaining)
        current_len = len(current)
        
        for _ in range(max_iters):
            if time.time() - start_time > DEADLINE:
                break
            
            improved = False
            
            # Try removing each element and adding two
            for remove_idx in range(len(current)):
                if time.time() - start_time > DEADLINE:
                    break
                
                removed_val = current[remove_idx]
                test_seq = current[:remove_idx] + current[remove_idx+1:]
                
                # Get available elements (original remaining + removed_val)
                avail = list(current_remaining) + [removed_val]
                
                # Try adding two elements
                for i, e1 in enumerate(avail):
                    if time.time() - start_time > DEADLINE:
                        break
                    for pos1 in range(len(test_seq) + 1):
                        # Check if e1 can be inserted at pos1
                        new_seq1 = test_seq[:pos1] + [e1] + test_seq[pos1:]
                        if not is_valid(new_seq1):
                            continue
                        
                        # Try adding a second element
                        for j, e2 in enumerate(avail):
                            if e2 == e1:
                                continue
                            if time.time() - start_time > DEADLINE:
                                break
                            for pos2 in range(len(new_seq1) + 1):
                                new_seq2 = new_seq1[:pos2] + [e2] + new_seq1[pos2:]
                                if is_valid(new_seq2) and len(new_seq2) > current_len:
                                    current = new_seq2
                                    current_len = len(new_seq2)
                                    # Update remaining
                                    current_remaining = [x for x in current_remaining if x != e1 and x != e2]
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        return current

    # Apply local search
    if time.time() - start_time < DEADLINE:
        used = set(best_seq)
        remaining = [x for x in range(2, 129) if x not in used]
        improved_seq = local_search_2move(best_seq, remaining)
        if is_valid(improved_seq) and len(improved_seq) > best_len:
            best_seq = improved_seq
            best_len = len(best_seq)

    # Additional local search passes with different initial states
    if time.time() - start_time < DEADLINE:
        # Try local search starting from greedy result
        greedy_seq = greedy_construct()
        used = set(greedy_seq)
        remaining = [x for x in range(2, 129) if x not in used]
        improved_seq = local_search_2move(greedy_seq, remaining)
        if is_valid(improved_seq) and len(improved_seq) > best_len:
            best_seq = improved_seq
            best_len = len(best_seq)

    # Final validation
    if not is_valid(best_seq):
        best_seq = greedy_construct()

    return best_seq
