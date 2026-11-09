<<<<<<< SEARCH
    def evaluate_delta(A, idx, old_val, new_val):
        """Compute the change in max(|A+A|, |A*A|) when replacing A[idx] with new_val.
        Returns (new_score, changed) where changed is True if the score actually changed."""
        # Elements affected: all pairs involving idx
        # We need to remove contributions of old_val and add contributions of new_val
        # For efficiency, track which sums/prods are unique to old_val or new_val
        
        # Build the current sum/prod multisets (counts)
        sum_counts = Counter()
        prod_counts = Counter()
        for i, j in pairs:
            sum_counts[A[i] + A[j]] += 1
            prod_counts[A[i] * A[j]] += 1
        
        # Remove contributions involving idx with old_val
        for i, j in pairs:
            if i == idx or j == idx:
                s = A[i] + A[j]
                p = A[i] * A[j]
                sum_counts[s] -= 1
                prod_counts[p] -= 1
                if sum_counts[s] == 0:
                    del sum_counts[s]
                if prod_counts[p] == 0:
                    del prod_counts[p]
        
        # Add contributions involving idx with new_val
        for i, j in pairs:
            if i == idx or j == idx:
                s = (new_val if i == idx else A[i]) + (new_val if j == idx else A[j])
                p = (new_val if i == idx else A[i]) * (new_val if j == idx else A[j])
                sum_counts[s] += 1
                prod_counts[p] += 1
        
        new_score = max(len(sum_counts), len(prod_counts))
        return new_score
=======

>>>>>>> REPLACE

