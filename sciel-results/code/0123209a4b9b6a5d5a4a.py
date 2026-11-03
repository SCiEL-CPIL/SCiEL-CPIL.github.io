    def local_search(codewords):
        # Initialize counts
        counts = [0] * total_words
        for c in codewords:
            for nb in neighbors[c]:
                counts[nb] += 1
                
        # Check validity
        for i in range(total_words):
            if counts[i] == 0:
                return codewords, counts # Invalid
                
        improved = True
        while improved:
            improved = False
            for c in list(codewords):
                # Check if c is redundant
                redundant = True
                for nb in neighbors[c]:
                    if counts[nb] <= 1:
                        redundant = False
                        break
                if redundant:
                    # Remove c
                    codewords.remove(c)
                    for nb in neighbors[c]:
                        counts[nb] -= 1
                    improved = True
                    break
                    
        # Try swaps
        for _ in range(20): # Limit swap iterations to prevent long stalls
            if not codewords:
                break
            c_out = random.choice(list(codewords))
            c_in = random.randint(0, total_words - 1)
            if c_in in codewords:
                continue
            
            # Simulate swap
            # We need to check if the new set is valid.
            # Valid if min count > 0 after swap.
            
            # Optimization: Only check bits affected by c_out and c_in
            # But counts are global.
            
            # Let's just do the copy check for simplicity, N=2187 is small.
            new_counts = counts[:]
            
            for nb in neighbors[c_out]:
                new_counts[nb] -= 1
            for nb in neighbors[c_in]:
                new_counts[nb] += 1
                
            valid = True
            for i in range(total_words):
                if new_counts[i] == 0:
                    valid = False
                    break
            
            if valid:
                codewords.remove(c_out)
                codewords.add(c_in)
                counts = new_counts
                # Restart removal phase
                # We can just break and let the outer while loop handle it?
                # No, the outer while loop checks `improved`.
                # Let's set improved = True and break the swap loop.
                improved = True
                break
                
        return codewords, counts
