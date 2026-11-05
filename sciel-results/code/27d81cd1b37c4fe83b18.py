import numpy as np


class Policy:
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.gamma = 0.92
        self.ema = np.zeros(n_arms)
        self.t = 0
        self.epsilon_init = 0.5
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.999
        
    def select(self, context):
        self.t += 1
        epsilon = max(self.epsilon_min, self.epsilon_init * (self.epsilon_decay ** self.t))
        
        if np.random.random() < epsilon:
            return int(np.random.randint(self.n))
        else:
            return int(np.argmax(self.ema))

    def update(self, context, arm, reward):
        r = reward
        self.ema[arm] = self.gamma * self.ema[arm] + (1 - self.gamma) * r
