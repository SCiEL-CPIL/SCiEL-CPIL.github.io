   class F9:
       def __init__(self, a, b):
           # a + b*i, i^2 = 2 (since -1 = 2 in F3)
           self.a = a % 3
           self.b = b % 3
       
       def __add__(self, other):
           return F9(self.a + other.a, self.b + other.b)
       
       def __sub__(self, other):
           return F9(self.a - other.a, self.b - other.b)
       
       def __mul__(self, other):
           # (a + bi)(c + di) = ac + adi + bci + bdi^2
           # i^2 = 2
           # = (ac + 2bd) + (ad + bc)i
           a = (self.a * other.a + 2 * self.b * other.b) % 3
           b = (self.a * other.b + self.b * other.a) % 3
           return F9(a, b)
       
       def __eq__(self, other):
           return self.a == other.a and self.b == other.b
       
       def __hash__(self):
           return hash((self.a, self.b))
       
       def inv(self):
           # 1 / (a+bi) = (a-bi) / (a^2 + b^2*i^2) = (a-bi) / (a^2 + 2b^2)
           denom = (self.a**2 + 2 * self.b**2) % 3
           if denom == 0:
               raise ValueError("Zero has no inverse")
           denom_inv = pow(denom, -1, 3) # 1^-1=1, 2^-1=2
           a = (self.a * denom_inv) % 3
           b = (-self.b * denom_inv) % 3
           return F9(a, b)
       
       def __repr__(self):
           return f"{self.a}+{self.b}i"
   