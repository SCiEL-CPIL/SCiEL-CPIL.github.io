import time
START_TIME = time.time()
def time_guard():
    if time.time() - START_TIME > 140:
        raise TimeoutError("Time limit exceeded")
