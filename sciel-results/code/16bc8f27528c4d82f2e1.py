    def reconstruct(weights, target):
        coeffs = [0] * len(weights)
        n_w = len(weights)
        # Precompute suffix sums of weights (all positive)
        suffix_sums = [0] * (n_w + 1)
        for i in range(n_w - 1, -1, -1):
            suffix_sums[i] = suffix_sums[i + 1] + weights[i]
        
        def bt(idx, cur_sum):
            if idx == n_w:
                return cur_sum == target
            
            # Pruning
            if abs(target - cur_sum) > suffix_sums[idx]:
                return False
                
            w = weights[idx]
            gap = target - cur_sum
            
            if gap > 0:
                vals = (1, 0, -1)
            elif gap < 0:
                vals = (-1, 0, 1)
            else:
                vals = (0, 1, -1)
                
            for val in vals:
                coeffs[idx] = val
                if bt(idx + 1, cur_sum + val * w):
                    return True
            coeffs[idx] = 0
            return False
        
        if bt(0, 0):
            return coeffs
        return None
