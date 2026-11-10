def construct():
    import math
    
    n = 512
    max_divs = 0
    best_n = 1
    
    # Find the number <= 512 with the maximum number of divisors
    for i in range(1, n + 1):
        divs = 0
        # Efficient divisor counting
        limit = int(math.isqrt(i))
        for j in range(1, limit + 1):
            if i % j == 0:
                if j * j == i:
                    divs += 1
                else:
                    divs += 2
        if divs > max_divs:
            max_divs = divs
            best_n = i
            
    # Return the set of divisors of best_n
    result = []
    for i in range(1, int(math.isqrt(best_n)) + 1):
        if best_n % i == 0:
            result.append(i)
            if i * i != best_n:
                result.append(best_n // i)
                
    return result
