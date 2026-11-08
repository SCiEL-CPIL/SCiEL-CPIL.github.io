<<<<<<< SEARCH
def generate_candidates(U, limit_val):
    """
    Generate candidate elements for addition based on sums and differences of existing elements.
    """
    U_set = set(U)
    candidates = set()
    
    for a in U:
        for b in U:
            s = a + b
            d = abs(a - b)
            if 0 < s <= limit_val and s not in U_set:
                candidates.add(s)
            if 0 < d <= limit_val and d not in U_set:
                candidates.add(d)
            
    return list(candidates)
=======
def generate_candidates(U, limit_val):
    """
    Generate candidate elements for addition based on sums and differences of existing elements.
    """
    U_set = set(U)
    candidates = set()
    
    for a in U:
        for b in U:
            s = a + b
            d = abs(a - b)
            if 0 < s <= limit_val and s not in U_set:
                candidates.add(s)
            if 0 < d <= limit_val and d not in U_set:
                candidates.add(d)
            
    return list(candidates)

def digit_based_search(U, deadline, max_val_limit=1000, max_n_limit=30):
    """
    Digit-based local search: parameterize elements via base-b digits and perturb digit choices.
    This explicitly controls additive structure by working in a mixed-radix representation.
    """
    U = sorted(set(U))
    current_score = score_set(U)
    best_score = current_score
    best_set = list(U)
    
    if len(U) < 2:
        return best_score, best_set
    
    max_u = max(U)
    if max_u <= 1:
        return best_score, best_set
    
    # Determine base and number of digits
    # Use base 2 or 3 for fine-grained control
    bases_to_try = [2, 3, 4]
    best_base = 2
    best_digits = []
    
    # Find a good base representation
    for base in bases_to_try:
        # Number of digits needed
        num_digits = int(floor(log(max_u) / log(base))) + 2 if max_u > 1 else 1
        
        # Try to represent each element in this base
        digits = []
        valid = True
        for u in U:
            d = []
            temp = u
            for _ in range(num_digits):
                d.append(temp % base)
                temp //= base
            if temp > 0:
                valid = False
                break
            digits.append(tuple(d))
        
        if valid:
            best_base = base
            best_digits = digits
            break
    
    if not best_digits:
        return best_score, best_set
    
    base = best_base
    num_digits = len(best_digits[0])
    
    no_improvement = 0
    max_no_improvement = 50
    
    while time.time() < deadline - 2 and no_improvement < max_no_improvement:
        improved = False
        
        # Randomly perturb one element's digits
        elem_idx = random.randint(1, len(U) - 1)  # Don't perturb 0
        digit_pos = random.randint(0, num_digits - 1)
        
        # Try changing this digit
        old_digit = best_digits[elem_idx][digit_pos]
        
        for new_digit in range(base):
            if new_digit == old_digit:
                continue
            
            # Apply change
            new_digits = list(best_digits[elem_idx])
            new_digits[digit_pos] = new_digit
            
            # Convert back to integer
            new_val = 0
            for i, d in enumerate(new_digits):
                new_val += d * (base ** i)
            
            if new_val <= 0 or new_val > max_val_limit:
                continue
            if new_val in U:
                continue
            
            # Try this change
            new_U = list(U)
            new_U[elem_idx] = new_val
            new_U = sorted(set(new_U))
            
            if len(new_U) != len(U):
                continue
            
            new_score = score_set(new_U)
            
            if new_score > current_score:
                # Accept and update digit representation
                U = new_U
                current_score = new_score
                
                # Re-derive digits for the changed element
                temp = new_val
                for i in range(num_digits):
                    best_digits[elem_idx] = best_digits[elem_idx][:i] + (temp % base,) + best_digits[elem_idx][i+1:]
                    temp //= base
                
                improved = True
                no_improvement = 0
                
                if current_score > best_score:
                    best_score = current_score
                    best_set = list(U)
                
                # Also try to adjust other digits of this element
                for dp in range(num_digits):
                    if dp == digit_pos:
                        continue
                    od = best_digits[elem_idx][dp]
                    for nd in range(base):
                        if nd == od:
                            continue
                        nd_list = list(best_digits[elem_idx])
                        nd_list[dp] = nd
                        nv = sum(d * (base ** i) for i, d in enumerate(nd_list))
                        if nv <= 0 or nv > max_val_limit or nv in U:
                            continue
                        nu = list(U)
                        nu[elem_idx] = nv
                        nu = sorted(set(nu))
                        if len(nu) != len(U):
                            continue
                        ns = score_set(nu)
                        if ns > current_score:
                            U = nu
                            current_score = ns
                            temp = nv
                            for i2 in range(num_digits):
                                best_digits[elem_idx] = best_digits[elem_idx][:i2] + (temp % base,) + best_digits[elem_idx][i2+1:]
                                temp //= base
                            no_improvement = 0
                            if current_score > best_score:
                                best_score = current_score
                                best_set = list(U)
                            break
                break
        
        # Try adding a new element with random digits
        if not improved and len(U) < max_n_limit:
            for _ in range(5):
                new_digits = tuple(random.randint(0, base - 1) for _ in range(num_digits))
                new_val = sum(d * (base ** i) for i, d in enumerate(new_digits))
                if new_val <= 0 or new_val > max_val_limit or new_val in U:
                    continue
                
                new_U = sorted(U + [new_val])
                new_score = score_set(new_U)
                
                if new_score > current_score:
                    U = new_U
                    current_score = new_score
                    improved = True
                    no_improvement = 0
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_set = list(U)
                    
                    # Add digits for new element
                    best_digits.append(new_digits)
                    break
        
        # Try removing an element
        if not improved and len(U) > 3:
            for _ in range(3):
                rem_idx = random.randint(1, len(U) - 1)
                new_U = [U[i] for i in range(len(U)) if i != rem_idx]
                new_score = score_set(new_U)
                
                if new_score > current_score:
                    U = new_U
                    current_score = new_score
                    improved = True
                    no_improvement = 0
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_set = list(U)
                    
                    best_digits.pop(rem_idx)
                    break
        
        if not improved:
            no_improvement += 1
    
    return best_score, best_set
>>>>>>> REPLACE

