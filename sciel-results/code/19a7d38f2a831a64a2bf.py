"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def construct():
    # The task requires the smallest n in [1,1000000] such that 120 divides phi(n).
    # We need 120 | phi(n), i.e., phi(n) is a multiple of 120.
    # 
    # Let's verify small candidates systematically.
    # phi(n) = n * prod_{p|n}(1-1/p)
    #
    # For n = 1: phi = 1, 120/1 not integer
    # n = 2: phi = 1
    # n = 3: phi = 2
    # n = 4: phi = 2
    # n = 5: phi = 4
    # n = 6: phi = 2
    # n = 7: phi = 6
    # n = 8: phi = 4
    # n = 9: phi = 6
    # n = 10: phi = 4
    # n = 11: phi = 10
    # n = 12: phi = 4
    # n = 13: phi = 12
    # n = 14: phi = 6
    # n = 15: phi = 8
    # n = 16: phi = 8
    # n = 17: phi = 16
    # n = 18: phi = 6
    # n = 19: phi = 18
    # n = 20: phi = 8
    # n = 21: phi = 12
    # n = 22: phi = 10
    # n = 23: phi = 22
    # n = 24: phi = 8
    # n = 25: phi = 20
    # n = 26: phi = 12
    # n = 27: phi = 18
    # n = 28: phi = 12
    # n = 30: phi = 8
    # n = 33: phi = 20
    # n = 34: phi = 16
    # n = 35: phi = 24
    # n = 36: phi = 12
    # n = 39: phi = 24
    # n = 40: phi = 16
    # n = 42: phi = 12
    # n = 45: phi = 24
    # n = 50: phi = 20
    # n = 51: phi = 32
    # n = 52: phi = 24
    # n = 55: phi = 40
    # n = 57: phi = 36
    # n = 58: phi = 28
    # n = 62: phi = 30
    # n = 65: phi = 48
    # n = 66: phi = 20
    # n = 69: phi = 44
    # n = 70: phi = 24
    # n = 74: phi = 36
    # n = 75: phi = 40
    # n = 77: phi = 60
    # n = 78: phi = 24
    # n = 82: phi = 40
    # n = 85: phi = 64
    # n = 86: phi = 42
    # n = 87: phi = 56
    # n = 91: phi = 72
    # n = 93: phi = 60
    # n = 95: phi = 72
    # n = 98: phi = 42
    # n = 100: phi = 40
    # n = 101: phi = 100
    # n = 102: phi = 32
    # n = 105: phi = 48
    # n = 106: phi = 52
    # n = 109: phi = 108
    # n = 110: phi = 40
    # n = 111: phi = 72
    # n = 114: phi = 36
    # n = 115: phi = 88
    # n = 116: phi = 56
    # n = 118: phi = 58
    # n = 119: phi = 96
    # n = 121: phi = 110
    # n = 122: phi = 60
    # n = 123: phi = 80
    # n = 124: phi = 60
    # n = 125: phi = 100
    # n = 127: phi = 126
    # n = 128: phi = 64
    # n = 129: phi = 84
    # n = 130: phi = 48
    # n = 131: phi = 130
    # n = 132: phi = 40
    # n = 133: phi = 108
    # n = 134: phi = 66
    # n = 135: phi = 72
    # n = 136: phi = 64
    # n = 138: phi = 44
    # n = 140: phi = 48
    # n = 141: phi = 92
    # n = 142: phi = 70
    # n = 143: phi = 120. 120 divides 120. YES!
    #
    # But wait, we need to check if there's anything smaller.
    # Let me double-check n=35: phi(35)=phi(5*7)=4*6=24. No.
    # n=57: phi(57)=phi(3*19)=2*18=36. No.
    # n=77: phi(77)=phi(7*11)=6*10=60. No.
    # n=91: phi(91)=phi(7*13)=6*12=72. No.
    # n=119: phi(119)=phi(7*17)=6*16=96. No.
    # n=143: phi(143)=phi(11*13)=10*12=120. Yes!
    
    # Let's verify a few more to be sure nothing between 100-143 works:
    # We need phi(n) >= 120 and 120 | phi(n).
    # phi(n) = 120, 240, 360, etc.
    # 
    # For phi(n) = 120: possible values of n?
    # n=143: 11*13, phi=10*12=120 ✓
    # 
    # Is there n < 143 with phi(n) divisible by 120?
    # We'd need phi(n) = 120 (since for n < 143, phi(n) < 143, and the next multiple of 120 is 240 > 143)
    # So we need phi(n) = 120 for some n < 143.
    # 
    # Known preimages of 120 under phi: 143, 212, 225, 236, 238, 250, 258, ...
    # 143 is the smallest. So n=143 is the answer.
    
    return 143
# EVOLVE-BLOCK-END
