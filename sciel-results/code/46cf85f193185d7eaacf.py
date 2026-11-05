import random

def _square_nonzero_count(coeffs):
    if not coeffs:
        return 0
    square = {}
    n = len(coeffs)
    for i in range(n):
        for j in range(i, n):
            e = coeffs[i][0] + coeffs[j][0]
            c = coeffs[i][1] * coeffs[j][1]
            if i == j:
                square[e] = square.get(e, 0) + c
            else:
                square[e] = square.get(e, 0) + 2 * c
    return sum(1 for v in square.values() if v != 0)

def construct():
    random.seed(12345)
    
    best_count = float('inf')
    best_coeffs = None
    
    # Multiple restarts
    for trial in range(8):
        # Start with a structured random polynomial
        exponents = sorted(random.sample(range(0, 80), 12))
        coeffs = [(e, random.choice([-1, 1])) for e in exponents]
        
        current_count = _square_nonzero_count(coeffs)
        
        # Local search with occasional larger moves
        for step in range(3000):
            move_type = random.random()
            
            if move_type < 0.7:
                # Change a coefficient
                idx = random.randint(0, 11)
                old_c = coeffs[idx][1]
                new_c = random.choice([-2, -1, 1, 2])
                if new_c == old_c:
                    continue
                coeffs[idx] = (coeffs[idx][0], new_c)
                new_count = _square_nonzero_count(coeffs)
                if new_count >= current_count:
                    coeffs[idx] = (coeffs[idx][0], old_c)
                else:
                    current_count = new_count
                    if current_count < best_count:
                        best_count = current_count
                        best_coeffs = list(coeffs)
            else:
                # Change an exponent slightly
                idx = random.randint(0, 11)
                old_e = coeffs[idx][0]
                new_e = old_e + random.choice([-1, 1])
                if new_e < 0:
                    continue
                # Check duplicate
                if any(coeffs[i][0] == new_e for i in range(12) if i != idx):
                    continue
                coeffs[idx] = (new_e, coeffs[idx][1])
                coeffs.sort(key=lambda x: x[0])
                new_count = _square_nonzero_count(coeffs)
                if new_count >= current_count:
                    coeffs[idx] = (old_e, coeffs[idx][1])
                    coeffs.sort(key=lambda x: x[0])
                else:
                    current_count = new_count
                    if current_count < best_count:
                        best_count = current_count
                        best_coeffs = list(coeffs)
        
        if best_count <= 6:
            break
    
    if best_coeffs is None:
        return [1] * 12
    
    # Build result
    max_deg = max(e for e, c in best_coeffs)
    if max_deg >= 41:
        return [1] * 12
    
    result = [0] * (max_deg + 1)
    for e, c in best_coeffs:
        if c != 0:
            result[e] = c
    
    # Trim
    while result and result[-1] == 0:
        result.pop()
    
    if not result or result[-1] == 0:
        return [1] * 12
    
    # Verify constraints
    if len(result) > 41:
        return [1] * 12
    if any(abs(c) > 64 for c in result):
        return [1] * 12
    if sum(1 for c in result if c != 0) != 12:
        return [1] * 12
    
    return result
