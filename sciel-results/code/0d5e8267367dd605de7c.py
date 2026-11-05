import time
import sys

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin before 90s kill

    # The domain is integers in [2, 128]
    candidates = list(range(2, 129))
    
    # Beam Search State: (sequence, set_of_products)
    # We want to maximize the length of the sequence.
    # Initial state: empty sequence, empty product set.
    # However, empty sequence has no products. 
    # We start by adding the first element.
    
    # To handle the "singleton products are included" and "nonempty consecutive block" rule:
    # If sequence is [a, b, c], products are:
    # a, b, c (singletons)
    # a*b, b*c (pairs)
    # a*b*c (triplet)
    
    # We maintain a beam of the best partial sequences.
    # "Best" is primarily defined by length, but we might want to keep diverse
    # product sets to avoid dead ends.
    # Actually, since we just want the longest valid sequence, and all numbers are > 1,
    # longer sequences are always preferred if they are valid.
    # But a shorter sequence might allow a longer extension later?
    # No, if A is a prefix of B, and B is valid, A is valid.
    # But if A and B are different sequences of the same length, one might allow
    # more future additions than the other.
    # So we keep a beam of top-K sequences based on length, and potentially diversity.
    
    # Let's use a beam width. Since the search space is small (127 items),
    # but the branching factor is high, we need to be careful.
    # However, the constraint is very tight.
    
    # Optimization:
    # Instead of storing the full set of products for every beam candidate (which is expensive),
    # we can store the sequence and recompute products when checking validity for a new element.
    # But checking validity for a new element $x$ appended to sequence $S$:
    # New products are: $x$, $S[-1]*x$, $S[-2]*S[-1]*x$, ..., $prod(S)*x$.
    # We need to check if any of these are in the existing product set of $S$.
    # AND we need to ensure the new products are distinct from each other (they are, since S is increasing and >1).
    # AND we need to ensure they don't collide with products of other blocks?
    # Wait, the constraint is "Products of every nonempty consecutive block must be distinct".
    # This means ALL blocks in the FINAL sequence must have distinct products.
    # If we append $x$ to $S$, the new blocks are those ending at $x$.
    # The old blocks are those in $S$.
    # We must ensure:
    # 1. New blocks are distinct from each other. (Always true for increasing integers > 1)
    # 2. New blocks are distinct from old blocks.
    
    # So the state needs to carry the set of products of the current sequence.
    # For a sequence of length $L$, there are $L(L+1)/2$ products.
    # Max length is likely around 20-30?
    # $30 \times 31 / 2 = 465$ products.
    # Storing a set of 465 integers is cheap.
    # Storing a beam of 100 such states is also cheap.
    
    beam_width = 200
    # Initial beam: all singletons
    # But we can start with an empty beam and add the first element.
    # Or just start with all singletons as the first layer.
    
    # Let's start with all singletons
    beam = []
    for x in candidates:
        beam.append(([x], {x}))
    
    # We iterate through the candidates in increasing order?
    # No, the sequence must be strictly increasing.
    # So if we pick $x$, the next element must be $> x$.
    # This suggests we can process candidates in increasing order and maintain beams
    # for "sequences ending with value < current candidate".
    
    # Alternative:
    # Since the sequence must be strictly increasing, the order is fixed by the set.
    # We are effectively choosing a subset.
    # But the "consecutive block" constraint depends on the order.
    # If we sort the subset, the order is fixed.
    # So we can just do a beam search over subsets, sorted.
    
    # Let's refine the beam search:
    # We process numbers $x$ from 2 to 128.
    # For each $x$, we try to append it to any sequence in the beam that ends with a value $< x$.
    # If the new sequence is valid, we add it to a new beam for the next iteration.
    # We keep the top $beam_width$ sequences based on length (and maybe some heuristic).
    
    # Actually, a simpler approach:
    # The sequence is strictly increasing.
    # Let $S$ be the current sequence.
    # We iterate $x$ from $max(S)+1$ to 128.
    # This is a depth-first search with pruning, or a beam search over the last element.
    
    # Let's use a Beam Search where the state is the sequence itself.
    # We keep the top K sequences by length.
    # To generate successors, we try appending the next possible integer.
    
    # Wait, if we just try to append the next integer, we might miss a better combination
    # where we skip an integer to fit a larger one later?
    # No, the goal is to maximize LENGTH.
    # Skipping an integer reduces the pool of available integers.
    # Generally, including more smaller numbers is better for length, unless they cause collisions.
    
    # Let's try a Beam Search that explores all possible next steps (any integer > last element).
    # State: (sequence, product_set)
    # Successors: for each $y > seq[-1]$, if valid, add $y$.
    
    # This branching factor is up to 127.
    # Beam width 200.
    # Depth ~20.
    # Total nodes ~ $200 \times 127 \times 20 \approx 500,000$.
    # Each check involves computing ~20 new products and checking membership in a set of ~200 products.
    # This is fast enough.
    
    def is_valid_append(seq, prod_set, new_val):
        # Check if appending new_val to seq is valid
        # New products: 
        # 1. new_val
        # 2. seq[-1] * new_val
        # 3. seq[-2] * seq[-1] * new_val
        # ...
        # k. prod(seq) * new_val
        
        # We need to check if any of these are in prod_set.
        # Also, are the new products distinct from each other?
        # Yes, because seq is strictly increasing and > 1.
        # $P_i = (\prod_{j=L-i+1}^{L} seq[j]) \times new\_val$
        # Since $seq[j] \ge 2$, the prefix products are strictly increasing.
        # So the new products are distinct.
        
        # So we only need to check against existing prod_set.
        
        # Optimization: 
        # If $new\_val$ is in prod_set, invalid.
        # If $seq[-1] * new\_val$ is in prod_set, invalid.
        # etc.
        
        # Let's compute the new products and check.
        # We can break early if a collision is found.
        
        # Also, we must add the new products to the set for the next step.
        # But we just return True/False here.
        
        # One more check: 
        # The problem says "Products of every nonempty consecutive block must be distinct".
        # This implies global distinctness.
        
        # Implementation:
        # Generate new products.
        # If any is in prod_set, return False.
        # Else return True.
        
        # To be safe and fast, we compute the products in reverse order (from the tail).
        # Actually, we just need to check intersection.
        
        # Let's do it simply.
        new_products = []
        prod = new_val
        new_products.append(prod)
        
        # Multiply by elements of seq from right to left
        # seq = [a, b, c]
        # new products: new_val, c*new_val, b*c*new_val, a*b*c*new_val
        
        # We can iterate seq in reverse.
        for a in reversed(seq):
            prod *= a
            new_products.append(prod)
            if prod in prod_set:
                return False
        
        # Check if new_val is in prod_set
        if new_val in prod_set:
            return False
            
        # We should also check if any new_product is in prod_set
        # The loop above checks `prod` (which accumulates) against `prod_set`.
        # But `prod` starts as `new_val`.
        # So the first check `if prod in prod_set` inside the loop covers `new_val`?
        # No, the loop starts with `a` being the last element of seq.
        # `prod` becomes `seq[-1] * new_val`.
        # So `new_val` itself is NOT checked in the loop if we don't check it before.
        
        # Let's re-structure:
        # 1. Check new_val in prod_set.
        # 2. Check suffix products * new_val in prod_set.
        
        if new_val in prod_set:
            return False
            
        suffix_prod = new_val
        for a in reversed(seq):
            suffix_prod *= a
            if suffix_prod in prod_set:
                return False
                
        return True

    # Beam Search
    # Initial beam: all singletons
    # But we can be smarter. 
    # Start with empty sequence?
    # If we start with empty, we add the first element.
    # Let's just start with all singletons as the initial beam.
    
    beam = []
    for x in candidates:
        beam.append(([x], {x}))
        
    # We will iterate. In each iteration, we expand the beam.
    # To avoid infinite loops or re-processing, we note that the sequence is strictly increasing.
    # So the "depth" is bounded by 127.
    # But we can just run until we can't improve or time runs out.
    
    # Actually, a better structure for Beam Search on sequences:
    # The state is the sequence.
    # We expand each state by trying all valid next numbers.
    # Then we prune to the top K states.
    
    # What is the "score" for pruning?
    # Primary: Length.
    # Secondary: We might want to keep sequences that leave more room?
    # Hard to quantify. Let's just keep the longest ones.
    # If there are ties, we might want diversity.
    # For now, just length.
    
    best_seq = beam[0][0]
    best_len = 1
    
    # We can stop expanding if the maximum possible length is reached?
    # Max length is 127.
    
    # To speed up, we can sort the beam by length descending before expansion?
    # No, we expand ALL states in the beam.
    
    # Let's use a priority queue or just a list.
    # List is fine for width 200.
    
    # We need to handle the time limit.
    
    while time.time() - start_time < DEADLINE:
        next_beam = []
        
        for seq, prod_set in beam:
            last_val = seq[-1]
            # Try all y > last_val
            # Optimization: if len(seq) + (128 - last_val) <= best_len, we can't beat best_len.
            # But we might find a sequence of the same length with better properties?
            # No, we just want max length.
            # If we can't beat best_len, we don't need to explore this branch further
            # UNLESS we are looking for a different sequence of the same length?
            # The task is to "Maximize length". Any sequence of max length is valid.
            # So if we found a sequence of length L, we don't need to find another of length L.
            # We only care about L+1.
            
            # So, pruning:
            # Max possible length from this state = len(seq) + (128 - last_val)
            # If this <= best_len, skip expansion.
            
            max_possible = len(seq) + (128 - last_val)
            if max_possible <= best_len:
                continue
                
            for y in range(last_val + 1, 129):
                if is_valid_append(seq, prod_set, y):
                    new_seq = seq + [y]
                    # Update product set
                    # We can copy the set and add new products
                    # Or we can just store the seq and recompute products in the next step?
                    # Recomputing is $O(L^2)$. Updating is $O(L)$.
                    # Let's update.
                    
                    new_prod_set = set(prod_set)
                    suffix_prod = y
                    new_prod_set.add(suffix_prod)
                    for a in reversed(seq):
                        suffix_prod *= a
                        new_prod_set.add(suffix_prod)
                        
                    next_beam.append((new_seq, new_prod_set))
                    
                    if len(new_seq) > best_len:
                        best_len = len(new_seq)
                        best_seq = new_seq
        
        if not next_beam:
            break
            
        # Prune next_beam to beam_width
        # Sort by length descending
        next_beam.sort(key=lambda x: len(x[0]), reverse=True)
        
        # If the best length in next_beam is less than or equal to best_len,
        # and we've already found best_len, do we need to keep them?
        # Only if they can lead to best_len + 1.
        # But we already prune based on max_possible in the next iteration.
        
        # However, keeping a diverse set might help.
        # For now, just take the top beam_width.
        
        beam = next_beam[:beam_width]
        
        # If the top sequence in the beam has length <= best_len, 
        # and no sequence in the beam can exceed best_len (checked in next loop),
        # we might be stuck.
        
        # Optimization: If all sequences in the beam have max_possible <= best_len, break.
        # This is handled at the start of the loop.
        
    # Final check: ensure best_seq is valid
    # It should be, by construction.
    
    return best_seq

# Test locally
if __name__ == "__main__":
    import time
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Length: {len(res)}")
    print(f"Time: {end - start:.2f}s")
    print(res[:10])
    
    # Verify
    def verify(seq):
        prods = set()
        n = len(seq)
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= seq[j]
                if p in prods:
                    return False
                prods.add(p)
        return True
    print("Valid:", verify(res))
