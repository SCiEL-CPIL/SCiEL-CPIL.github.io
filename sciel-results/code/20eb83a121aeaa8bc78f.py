"""Divisibility-antichain representations by smooth numbers.
Global precomputation of minimum-cardinality antichain sums via backtracking."""

def construct():
    # Precompute 2^a * 3^b * 5^c smooth numbers up to 128
    smooth = []
    for a in range(0, 8):
        p2 = 2 ** a
        if p2 > 128:
            break
        for b in range(0, 6):
            p3 = 3 ** b
            if p2 * p3 > 128:
                break
            for c in range(0, 5):
                p5 = 5 ** c
                val = p2 * p3 * p5
                if val > 128:
                    break
                smooth.append(val)
    smooth = sorted(set(smooth))
    
    # best_rep[n] stores the tuple of terms for the minimum cardinality antichain summing to n
    best_rep = {}
    
    def generate_antichains(k):
        def backtrack(start_idx, current_set, current_sum):
            if len(current_set) == k:
                if current_sum <= 128:
                    if current_sum not in best_rep:
                        best_rep[current_sum] = tuple(current_set)
                return
            
            remaining_needed = k - len(current_set)
            if len(smooth) - start_idx < remaining_needed:
                return
                
            for i in range(start_idx, len(smooth)):
                val = smooth[i]
                
                # Pruning: if adding val exceeds 128, skip
                if current_sum + val > 128:
                    continue
                
                # Check antichain property with existing elements
                compatible = True
                for existing in current_set:
                    if val % existing == 0 or existing % val == 0:
                        compatible = False
                        break
                if not compatible:
                    continue
                
                current_set.append(val)
                backtrack(i + 1, current_set, current_sum + val)
                current_set.pop()

        backtrack(0, [], 0)

    # Generate antichains for increasing k until all targets 1..128 are covered
    for k in range(1, 31):
        generate_antichains(k)
        if all(n in best_rep for n in range(1, 129)):
            break
    
    answers = []
    for n in range(1, 129):
        if n in best_rep:
            answers.append(list(best_rep[n]))
        else:
            answers.append([])
            
    return answers
