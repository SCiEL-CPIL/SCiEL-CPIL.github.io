import time
from math import factorial

def construct():
    # Hardcoded solution: 33 * 28 * 25 * 32 * 27 * 24 = 479001600 = 12!
    # Verify: 33=3*11, 28=4*7=2^2*7, 25=5^2, 32=2^5, 27=3^3, 24=8*3=2^3*3
    # Product: 3*11 * 2^2*7 * 5^2 * 2^5 * 3^3 * 2^3*3 = 2^(2+5+3) * 3^(1+3+1) * 5^2 * 7 * 11
    # = 2^10 * 3^5 * 5^2 * 7 * 11 = 12!
    # All distinct, all > 12, count = 6 <= 12, max = 33
    return [33, 28, 25, 32, 27, 24]
