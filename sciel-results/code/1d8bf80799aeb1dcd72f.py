import time

def construct():
    # Valid set based on 3*min(S) > 2*max(S)
    # 3*K > 2*256 => K > 170.66 => K=171
    # S = {171, ..., 256}
    # Size = 86
    return list(range(171, 257))
