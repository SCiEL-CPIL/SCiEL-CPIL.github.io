import math

def construct():
    n = 255
    # Prime factors of 255 are 3, 5, 17
    # We want to maximize the number of multiples in [1, 255]
    # which is floor(255 / p)
    # This is maximized for the smallest prime factor p.
    
    # Find smallest prime factor
    def smallest_prime_factor(num):
        if num % 2 == 0:
            return 2
        if num % 3 == 0:
            return 3
        if num % 5 == 0:
            return 5
        if num % 7 == 0:
            return 7
        if num % 11 == 0:
            return 11
        if num % 13 == 0:
            return 13
        if num % 17 == 0:
            return 17
        # For robustness, though 255 is fixed
        i = 19
        while i * i <= num:
            if num % i == 0:
                return i
            i += 2
        return num

    p = smallest_prime_factor(n)
    return list(range(p, n + 1, p))
