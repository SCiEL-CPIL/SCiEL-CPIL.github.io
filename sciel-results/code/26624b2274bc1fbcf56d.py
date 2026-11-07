import math
import time

def construct():
    # The best known solution is {224, 240}
    # Max is 240.
    # 210 * 224 * 240 = 210 * 53760 = 11289600
    # sqrt(11289600) = 3360
    # 3360^2 = 11289600
    # 224 > 210, 240 > 210
    # Distinct.
    # Return as list of ints.
    return [224, 240]
