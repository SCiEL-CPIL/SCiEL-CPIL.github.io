import math

def construct():
    # Precompute smallest prime factors up to 50000
    # Max B = n*x. n=256, x=192. B=49152.
    LIMIT = 50000
    spf = list(range(LIMIT + 1))
    for i in range(2, int(LIMIT ** 0.5) + 1):
        if spf[i] == i:
            for j in range(i * i, LIMIT + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def get_divisors_of_square(n):
        # Factorize n
        factors = {}
        temp = n
        while temp > 1:
            p = spf[temp]
            count = 0
            while temp % p == 0:
                temp //= p
                count += 1
            factors[p] = count
        # Generate divisors of n^2
        # If n = p1^a1 ... pk^ak, n^2 = p1^(2a1) ...
        divs = [1]
        for p, a in factors.items():
            exp = 2 * a
            current_divs = []
            p_pow = 1
            for e in range(exp + 1):
                for d in divs:
                    current_divs.append(d * p_pow)
                p_pow *= p
            divs = current_divs
        return divs

    answers = []
    bound = 1000000000000
    for n in range(3, 257):
        found = None
        x_start = n // 4 + 1
        x_end = (3 * n) // 4
        
        for x in range(x_start, x_end + 1):
            A = 4 * x - n
            B = n * x
            B2 = B * B
            U_min = 4 * x * x - 2 * n * x
            
            # Get divisors of B^2
            divs = get_divisors_of_square(B)
            
            # We need U < B (so U < V) and U > U_min
            # And (U+B) % A == 0
            for U in divs:
                if U >= B:
                    continue
                if U <= U_min:
                    continue
                
                if (U + B) % A != 0:
                    continue
                y = (U + B) // A
                if y <= x or y > bound:
                    continue
                
                V = B2 // U
                if (V + B) % A != 0:
                    continue
                z = (V + B) // A
                if z <= y or z > bound:
                    continue
                
                found = [x, y, z]
                break
            if found:
                break
        answers.append(found)
    return answers
