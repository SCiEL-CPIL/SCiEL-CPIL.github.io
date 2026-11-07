import time
import sys

def construct():
    n = 10
    
    # Generate L1 using simple addition
    L1 = [[0 for _ in range(n)] for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            L1[i][j] = (i + j) % n
            
    return [L1]
